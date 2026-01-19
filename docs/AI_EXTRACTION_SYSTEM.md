# Sistema Agentico per Estrazione Metadata con AI

## Panoramica

PyPotteryLens utilizza un sistema di estrazione metadata basato su AI con un approccio **agentico a due passaggi** per massimizzare l'accuratezza nell'identificazione di periodi, figure e descrizioni delle ceramiche archeologiche.

## Architettura a Due Passaggi

### Passaggio 1: Analisi Struttura Documento

Il `DocumentStructureAnalyzer` analizza l'**intero PDF** per costruire una mappa globale delle relazioni tra figure/tavole e periodi cronologici.

```
Input: Intero testo PDF

L'AI costruisce mappature come:

  Tafel/Figure → Periodo
  ─────────────────────────
  "Tafel 1-8"    → "Umm an-Nar"
  "Tafel 9-15"   → "Wadi Suq"
  "Abb. 174"     → "Iron Age II"

  Catalog ID → Periodo
  ─────────────────────────
  "BAT10A-0177"  → "Umm an-Nar"
  "M5-12"        → "Late Bronze Age"

Output: DocumentStructure (mappature globali)
```

### Passaggio 2: Estrazione Per Immagine

Per **ogni immagine** di ceramica, l'estrattore riceve:

1. **Immagine** (base64) - il ritaglio PNG della ceramica
2. **Contesto PDF** - la caption/testo circostante
3. **Document Structure** - le mappature globali dal passaggio 1

```
┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐
│  IMMAGINE   │  │ CONTESTO PDF │  │ DOCUMENT STRUCTURE  │
│  (base64)   │  │ (caption)    │  │ (mappature globali) │
└──────┬──────┘  └──────┬───────┘  └──────────┬──────────┘
       │                │                      │
       └────────────────┼──────────────────────┘
                        ↓
                  ┌───────────┐
                  │    AI     │  ← Modello VISION
                  │  (LLaVA,  │    (vede immagine +
                  │   Claude, │     legge contesto)
                  │   GPT-4V) │
                  └─────┬─────┘
                        ↓
Output JSON:
{
  "figure_number": "Tafel 3, Fig. 2",
  "pottery_id": "BAT10A-0177",
  "period": "Umm an-Nar",
  "description": "Globular jar with geometric decoration..."
}
```

## Perché Servono Modelli VISION?

L'AI riceve **due tipi di informazione**:

### 1. Immagine Ceramica (PNG)

L'AI deve **VEDERE** l'immagine per:
- Identificare la forma (vaso, ciotola, anfora)
- Riconoscere decorazioni (geometriche, dipinte)
- Valutare lo stato di conservazione (integro, frammentario)
- Descrivere caratteristiche morfologiche

### 2. Testo Contesto (PDF caption)

L'AI deve **LEGGERE** il contesto per:
- Estrarre numeri di figura/tavola
- Identificare ID di catalogo
- Trovare riferimenti cronologici

### Conseguenza

| Tipo Modello | Vede Immagine | Legge Testo | Funziona? |
|--------------|---------------|-------------|-----------|
| Solo Testo (es. Llama 3) | ❌ | ✅ | ❌ Parziale |
| Vision (es. LLaVA) | ✅ | ✅ | ✅ Completo |

**Un modello SOLO TESTO non può vedere l'immagine!** Può solo elaborare il contesto testuale, perdendo informazioni cruciali sulla ceramica stessa.

## Provider AI Supportati

### Cloud Providers (API Key richiesta)

| Provider | Modello Default | Vision | Note |
|----------|-----------------|--------|------|
| **Anthropic** | Claude Sonnet 4.5 | ✅ | Eccellente per archeologia |
| **OpenAI** | GPT-4.1 | ✅ | Ottima accuratezza |
| **Gemini** | Gemini 2.0 Flash | ✅ | Veloce e preciso |
| **DeepSeek** | DeepSeek Chat | ❌ | Solo testo, usa contesto |
| **Together AI** | Llama-Vision-Free | ✅ | **Gratuito!** |

### Local Providers (Gratuiti, esecuzione locale)

| Provider | Modello Consigliato | RAM Richiesta |
|----------|---------------------|---------------|
| **Ollama** | llava | ~8 GB |
| **Ollama** | llava:13b | ~16 GB |
| **LM Studio** | llava-v1.6-mistral-7b | ~8 GB |

## Installazione Ollama

### macOS

```bash
# 1. Installa Ollama
brew install ollama

# 2. Avvia il server
ollama serve

# 3. Scarica modello vision (4.1 GB)
ollama pull llava

# 4. (Opzionale) Modello più accurato (8 GB)
ollama pull llava:13b

# 5. Verifica installazione
ollama list
```

### Spostare modelli su disco esterno

```bash
# Ferma Ollama
pkill ollama

# Sposta la cartella
mv ~/.ollama /Volumes/TUO_DISCO/.ollama

# Crea symlink
ln -s /Volumes/TUO_DISCO/.ollama ~/.ollama

# Riavvia
ollama serve
```

### Configurazione in PyPotteryLens

1. Vai su **Tabular** → **AI Settings**
2. Seleziona **Ollama** come provider
3. Base URL: `http://localhost:11434` (default)
4. Model: `llava`

## Installazione LM Studio

### Download e Setup

1. Scarica da https://lmstudio.ai
2. Apri l'applicazione
3. Vai su **Discover** (icona lente)
4. Cerca "llava" o "vision"
5. Scarica un modello:
   - `llava-v1.6-mistral-7b` (~8 GB RAM) - Consigliato
   - `llava-v1.5-7b` (~6 GB RAM) - Più leggero
   - `qwen2-vl-7b-instruct` (~8 GB RAM) - Alternativa

### Avvio Server

1. Vai su **Local Server** (icona server)
2. Seleziona il modello vision dal dropdown
3. Clicca **Start Server**
4. Il server sarà disponibile su `http://localhost:1234/v1`

### Configurazione in PyPotteryLens

1. Vai su **Tabular** → **AI Settings**
2. Seleziona **LM Studio** come provider
3. Base URL: `http://localhost:1234/v1` (default)
4. Model: (lascia vuoto, usa quello caricato)

## Together AI (Cloud Gratuito)

Together AI offre accesso gratuito a modelli vision open-source.

### Setup

1. Registrati su https://together.ai
2. Ottieni la tua API key (gratuita)
3. In PyPotteryLens → **Tabular** → **AI Settings**
4. Seleziona **Together AI (LLaVA Free)**
5. Inserisci la tua API key

### Modello Default

`meta-llama/Llama-Vision-Free` - Modello vision gratuito con supporto immagini

## Flusso del Codice

```
app.py: ai_extract_metadata()
    │
    ├── 1. Carica PDF e estrai testo completo
    │       └── fitz.open(pdf_path) → get_text()
    │
    ├── 2. DocumentStructureAnalyzer.analyze()
    │       ├── Invia testo completo all'AI
    │       └── Costruisce DocumentStructure con mappature
    │
    ├── 3. Per ogni immagine card:
    │       │
    │       ├── get_extractor(provider, api_key, ...)
    │       │   └── Ritorna: ClaudeExtractor / OllamaExtractor / ...
    │       │
    │       ├── Carica immagine come base64
    │       │
    │       └── extractor.extract_metadata(
    │               image_base64,      # Immagine ceramica
    │               context,           # Testo PDF circostante
    │               document_structure # Mappature dal passo 1
    │           )
    │
    └── 4. Salva risultati in mask_info.csv
            └── figure_num, pottery_id, period, description
```

## File Principali

| File | Descrizione |
|------|-------------|
| `ai_extractor.py` | Classi estrattore per ogni provider |
| `app.py` | Endpoint `/api/projects/<id>/metadata/ai-extract` |
| `settings_manager.py` | Gestione API keys e configurazioni |

## Troubleshooting

### "0 successful" con Ollama/LM Studio

**Causa**: Modello solo testo caricato invece di vision

**Soluzione**: Assicurati di usare un modello vision:
- Ollama: `ollama pull llava`
- LM Studio: Cerca e scarica modelli con "llava" o "vision"

### "No models loaded" con LM Studio

**Causa**: Server avviato ma nessun modello selezionato

**Soluzione**: In LM Studio → Local Server → seleziona modello → Start Server

### Estrazione lenta

**Causa**: Modello troppo grande per la RAM disponibile

**Soluzione**:
- Usa modelli più piccoli (7B invece di 13B)
- Chiudi altre applicazioni
- Considera provider cloud

---

*Documentazione generata per PyPotteryLens v1.0*
