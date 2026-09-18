#include "server.h"
#include <iostream>
#include <sstream>
#include <cstring>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <csignal>

static volatile sig_atomic_t g_server_running = 1;

static void sigint_handler(int)
{
    g_server_running = 0;
}

void run_relic_interactive(InferenceEngine &engine, int default_max_tokens, float default_temp, int default_top_k)
{
    fprintf(stdout, "\n============================================================\n");
    fprintf(stdout, "       RELIC INTERACTIVE SESSION (VRAM PERSISTENT)          \n");
    fprintf(stdout, "============================================================\n");
    fprintf(stdout, "Model is 100%% resident in GPU memory. Ready for instant queries.\n");
    fprintf(stdout, "Commands: /exit, /quit, /n <tokens>, /temp <val>, /topk <val>, /reset\n");
    fprintf(stdout, "============================================================\n\n");

    int n_tokens = default_max_tokens;
    float temp = default_temp;
    int top_k = default_top_k;
    std::string line;

    while (true)
    {
        fprintf(stdout, ">>> ");
        fflush(stdout);
        if (!std::getline(std::cin, line))
            break;
        if (line.empty())
            continue;
        if (line == "/exit" || line == "/quit")
            break;
        if (line.rfind("/n ", 0) == 0)
        {
            n_tokens = atoi(line.c_str() + 3);
            fprintf(stdout, "Token count set to %d\n", n_tokens);
            continue;
        }
        if (line.rfind("/temp ", 0) == 0)
        {
            temp = (float)atof(line.c_str() + 6);
            fprintf(stdout, "Temperature set to %.2f\n", temp);
            continue;
        }
        if (line.rfind("/topk ", 0) == 0)
        {
            top_k = atoi(line.c_str() + 6);
            fprintf(stdout, "Top-k set to %d\n", top_k);
            continue;
        }
        if (line == "/reset")
        {
            engine.free_buffers();
            fprintf(stdout, "Context state / recurrent state reset.\n");
            continue;
        }

        fprintf(stdout, "\n");
        engine.generate(line, n_tokens, temp, top_k);
        fprintf(stdout, "\n\n");
        engine.free_buffers();
    }
    fprintf(stdout, "\nExiting interactive session. VRAM freed.\n");
}

static std::string extract_json_field(const std::string &json, const std::string &key)
{
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos)
        return "";
    pos += search.length();
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == ':' || json[pos] == '\t'))
        pos++;
    if (pos >= json.length())
        return "";

    if (json[pos] == '\"')
    {
        pos++;
        size_t end = json.find('\"', pos);
        if (end == std::string::npos)
            return "";
        return json.substr(pos, end - pos);
    }
    else
    {
        size_t end = pos;
        while (end < json.length() && json[end] != ',' && json[end] != '}' && json[end] != '\r' && json[end] != '\n')
            end++;
        return json.substr(pos, end - pos);
    }
}

static bool safe_write_all(int fd, const void *data, size_t len)
{
    const uint8_t *ptr = (const uint8_t *)data;
    while (len > 0)
    {
        ssize_t written = write(fd, ptr, len);
        if (written <= 0)
            return false;
        ptr += written;
        len -= (size_t)written;
    }
    return true;
}

static bool safe_write_str(int fd, const std::string &str)
{
    return safe_write_all(fd, str.data(), str.size());
}

bool run_relic_server(InferenceEngine &engine, int port, int default_max_tokens, float default_temp, int default_top_k)
{
    signal(SIGINT, sigint_handler);
    signal(SIGTERM, sigint_handler);

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0)
    {
        perror("socket failed");
        return false;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0)
    {
        perror("bind failed");
        close(server_fd);
        return false;
    }

    if (listen(server_fd, 10) < 0)
    {
        perror("listen failed");
        close(server_fd);
        return false;
    }

    fprintf(stdout, "\n============================================================\n");
    fprintf(stdout, "       RELIC PERSISTENT VRAM DAEMON ACTIVE                  \n");
    fprintf(stdout, "============================================================\n");
    fprintf(stdout, "Model is locked resident in GPU VRAM (Zero reload overhead).\n");
    fprintf(stdout, "Listening on http://127.0.0.1:%d\n", port);
    fprintf(stdout, "Endpoints:\n");
    fprintf(stdout, "  POST /generate  - Body: {\"prompt\": \"...\", \"n_tokens\": %d}\n", default_max_tokens);
    fprintf(stdout, "  GET  /health    - Server health check\n");
    fprintf(stdout, "  POST /reset     - Reset KV-cache & recurrent states\n");
    fprintf(stdout, "Query via CLI:    ./build/relic --client %d -p \"Your prompt\"\n", port);
    fprintf(stdout, "Query via curl:   curl -X POST http://127.0.0.1:%d/generate -d '{\"prompt\": \"Hello\"}'\n", port);
    fprintf(stdout, "============================================================\n\n");
    fflush(stdout);

    while (g_server_running)
    {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0)
        {
            if (!g_server_running)
                break;
            continue;
        }

        char buffer[16384];
        memset(buffer, 0, sizeof(buffer));
        ssize_t bytes_read = read(client_fd, buffer, sizeof(buffer) - 1);
        if (bytes_read <= 0)
        {
            close(client_fd);
            continue;
        }

        std::string req(buffer, bytes_read);

        if (req.find("GET /health") != std::string::npos)
        {
            std::string body = "{\"status\": \"ok\", \"vram_resident\": true, \"engine\": \"relic\"}\n";
            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " +
                               std::to_string(body.length()) + "\r\nConnection: close\r\n\r\n" + body;
            safe_write_str(client_fd, resp);
            close(client_fd);
            continue;
        }

        if (req.find("POST /reset") != std::string::npos)
        {
            engine.free_buffers();
            std::string body = "{\"status\": \"reset_complete\"}\n";
            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " +
                               std::to_string(body.length()) + "\r\nConnection: close\r\n\r\n" + body;
            safe_write_str(client_fd, resp);
            close(client_fd);
            continue;
        }

        // Parse prompt & parameters
        std::string prompt;
        int n_tokens = default_max_tokens;
        float temp = default_temp;
        int top_k = default_top_k;

        size_t body_pos = req.find("\r\n\r\n");
        std::string body = (body_pos != std::string::npos) ? req.substr(body_pos + 4) : req;

        if (body.find("{") != std::string::npos)
        {
            prompt = extract_json_field(body, "prompt");
            std::string tok_str = extract_json_field(body, "n_tokens");
            if (!tok_str.empty())
                n_tokens = atoi(tok_str.c_str());
            std::string temp_str = extract_json_field(body, "temperature");
            if (!temp_str.empty())
                temp = (float)atof(temp_str.c_str());
            std::string topk_str = extract_json_field(body, "top_k");
            if (!topk_str.empty())
                top_k = atoi(topk_str.c_str());
        }
        else
        {
            prompt = body;
            // Trim newlines
            while (!prompt.empty() && (prompt.back() == '\r' || prompt.back() == '\n'))
                prompt.pop_back();
        }

        if (prompt.empty())
            prompt = "The capital of France is";

        fprintf(stdout, "[Server Request] Prompt: \"%s\" (tokens: %d, temp: %.2f)\n", prompt.c_str(), n_tokens, temp);
        fflush(stdout);

        std::string generated = engine.generate(prompt, n_tokens, temp, top_k);
        engine.free_buffers();

        std::string resp_header = "HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: " +
                                  std::to_string(generated.length()) + "\r\nConnection: close\r\n\r\n";
        safe_write_str(client_fd, resp_header);
        safe_write_str(client_fd, generated);
        close(client_fd);
    }

    close(server_fd);
    fprintf(stdout, "\nRelic server stopped. GPU VRAM released.\n");
    return true;
}

int run_relic_client(int port, const std::string &prompt, int n_tokens, float temp, int top_k)
{
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0)
    {
        perror("client socket creation failed");
        return 1;
    }

    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);

    if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0)
    {
        fprintf(stderr, "Invalid address\n");
        close(sock);
        return 1;
    }

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
    {
        fprintf(stderr, "Could not connect to Relic persistent server at 127.0.0.1:%d.\n", port);
        fprintf(stderr, "Start the server first with: ./build/relic -m <model.gguf> --server %d\n", port);
        close(sock);
        return 1;
    }

    // Build JSON request
    std::string json_body = "{\"prompt\": \"" + prompt + "\", \"n_tokens\": " + std::to_string(n_tokens) +
                            ", \"temperature\": " + std::to_string(temp) + ", \"top_k\": " + std::to_string(top_k) + "}";

    std::string http_req = "POST /generate HTTP/1.1\r\n"
                           "Host: 127.0.0.1:" + std::to_string(port) + "\r\n"
                           "Content-Type: application/json\r\n"
                           "Content-Length: " + std::to_string(json_body.length()) + "\r\n"
                           "Connection: close\r\n\r\n" + json_body;

    safe_write_str(sock, http_req);

    char buffer[4096];
    std::string response;
    ssize_t bytes_read = 0;
    while ((bytes_read = read(sock, buffer, sizeof(buffer) - 1)) > 0)
    {
        response.append(buffer, bytes_read);
    }
    close(sock);

    size_t body_pos = response.find("\r\n\r\n");
    if (body_pos != std::string::npos)
    {
        fprintf(stdout, "%s\n", response.substr(body_pos + 4).c_str());
    }
    else
    {
        fprintf(stdout, "%s\n", response.c_str());
    }
    return 0;
}
