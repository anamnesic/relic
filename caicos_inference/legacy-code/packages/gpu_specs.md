# GPU alvo

## Identificacao

- Nome no Windows: AMD Radeon HD 7000 series
- PCI ID: `PCI\VEN_1002&DEV_6778`
- Vendor ID: `1002`
- Device ID: `6778`
- Chip exato: Caicos XT
- Nomes comerciais comuns: Radeon HD 7470/8470, R5 235/310 (OEM)

## Especificacoes consolidadas

- Familia: Radeon HD 7000 series / Southern Islands branding
- Arquitetura: TeraScale, com suporte legado da linha HD 7000 para chips nao-GCN
- Processo de fabricacao: 40 nm
- Transistores: 370 M
- Lancamento: janeiro de 2012
- Configuracao do chip: 160:8:4
- Unidades de shader: 160
- Texture Mapping Units: 8
- Render Output Units: 4
- Clock do core: 625 MHz base, 775 MHz boost
- Clock da memoria: 800 MHz a 900 MHz, dependendo da variante
- Interface de memoria: 64-bit
- Tipo de memoria: DDR3 ou GDDR5, conforme OEM
- VRAM observada nesta maquina: 1 GB
- Interface de barramento: PCIe 2.1 x16

## Recursos de plataforma

- OpenCL: suporte legado via driver; a serie HD 7000 aparece com suporte OpenCL, mas sem caminho moderno de ROCm para esse chip
- OpenGL: ate 4.5 no Windows e 4.6 no Linux para partes da serie GCN; Caicos segue o caminho legado TeraScale, entao nao entra na faixa moderna de Vulkan/ROCm
- Vulkan: nao suportado para TeraScale em pratica moderna
- Video: UVD e Eyefinity estao presentes na serie HD 7000
- Saidas: ate quatro outputs na familia, dependendo da placa e dos conectores

## Drivers e suporte

- Driver atual no Windows desta maquina: 15.201.1151.1008
- Driver Linux legado: `radeon` no kernel suporta a GPU; `fglrx` historicamente tambem suportou esse device
- Status geral: GPU antiga e fora do caminho moderno de ROCm

## Implicacoes para inferencia local

- Prioridade pratica: OpenCL e caminhos legados
- Expectativa realista: modelos muito pequenos e muito quantizados
- Restricao principal: 1 GB de VRAM e arquitetura antiga sem suporte moderno de treinamento/inferencia acelerada

## Fontes publicas usadas

- DeviceHunt para o nome do device e o ID PCI
- linux-hardware.org para o nome exato e informacoes de driver
- Wikipedia, pagina da Radeon HD 7000 series, para processo, transistores, clocks, memoria, cores, TMUs, ROPs e contexto da familia