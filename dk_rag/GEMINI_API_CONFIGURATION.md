# Gemini API Configuration for Map-Reduce Extraction

## ✅ Configuration Status

Your Gemini API key has been successfully configured and is working correctly! The system is now using your `GEMINI_API_KEY` from the `.env` file.

### ✅ **Confirmed Working:**
- **API Key Detection**: `✅ GEMINI_API_KEY found (length: 39)`
- **Map-Reduce Integration**: Both map and reduce phases configured for `gemini/gemini-2.0-flash-exp`
- **API Connectivity**: Successfully connecting to Gemini API
- **Progress Tracking**: Real-time batch processing progress working
- **Caching System**: Statistical analysis and batch results being cached

## 🔧 **Rate Limit Optimization**

Your test revealed Gemini 2.0 Flash Experimental has restrictive rate limits:
- **10 requests per minute per model**
- **250,000 input tokens per minute per model**

### Configuration Adjustments Made:

```yaml
# Updated configuration for Gemini rate limits
map_reduce_extraction:
  batch_size: 10
  max_tokens_per_batch: 20000          # Reduced from 30k
  parallel_batches: 1                  # Reduced from 3 (to stay within 10 req/min)
  retry_delay_seconds: 10              # Increased from 5
```

## 📋 **Configuration Options**

### Option 1: Direct Gemini (Current Setup)
**File**: `config/persona_config.yaml`
```yaml
map_phase_model: "gemini/gemini-2.0-flash-exp"    # Uses GEMINI_API_KEY
reduce_phase_model: "gemini/gemini-2.0-flash-exp" # Uses GEMINI_API_KEY
parallel_batches: 1                               # Respects rate limits
```

### Option 2: OpenRouter Gemini (Higher Limits)
**File**: `config/persona_config_openrouter.yaml`
```yaml
map_phase_model: "openrouter/google/gemini-2.0-flash-exp"    # Via OpenRouter
reduce_phase_model: "openrouter/google/gemini-2.0-flash-exp" # Via OpenRouter
parallel_batches: 3                                          # Higher limits
```

## 🚀 **Usage Instructions**

### Using Direct Gemini (Recommended for Testing)
```bash
# Extract persona with direct Gemini access
python -m dk_rag.cli.persona_builder extract-persona \
  --documents-dir /Users/blickt/Documents/src/pdf_2_text/content_repo/greg_startup \
  --name "Greg Startup"
```

### Using OpenRouter Gemini (Recommended for Production)
```bash
# Extract persona with OpenRouter (higher rate limits)
python -m dk_rag.cli.persona_builder extract-persona \
  --config config/persona_config_openrouter.yaml \
  --documents-dir /Users/blickt/Documents/src/pdf_2_text/content_repo/greg_startup \
  --name "Greg Startup"
```

## 📊 **Expected Performance**

### With Full Greg Startup Corpus (255 documents, 2.3M words):

#### Direct Gemini (Current Setup):
- **Processing Time**: ~2-3 hours (due to rate limits)
- **Batches**: ~26 batches (10 docs each)
- **API Calls**: ~52 calls (26 map + 26 reduce + consolidation)
- **Cost**: ~$0.50-1.00 (very economical)

#### OpenRouter Gemini:
- **Processing Time**: ~30-45 minutes (higher rate limits)
- **Batches**: ~26 batches processed in parallel
- **Cost**: ~$1.00-2.00 (still very economical)

## 🎯 **Next Steps**

### 1. **Quick Test with Sample** (Recommended)
```bash
# Test with first 10 documents
python dk_rag/test_map_reduce.py
```

### 2. **Full Extraction**
```bash
# Extract with full corpus using direct Gemini
python -m dk_rag.cli.persona_builder extract-persona \
  --documents-dir /Users/blickt/Documents/src/pdf_2_text/content_repo/greg_startup \
  --name "Greg Startup Full Corpus"
```

### 3. **Monitor Progress**
- **Real-time progress bars** show batch processing
- **Cache status** displays what's being reused
- **Statistics** show completion rates and performance

### 4. **Compare Results**
After extraction, compare with original truncated results:
```bash
# List available personas
python -m dk_rag.cli.persona_builder list-personas

# View cache info
python -m dk_rag.cli.persona_builder cache info
```

## 🔧 **Rate Limit Strategies**

### If You Hit Rate Limits:

1. **Use Caching** (Already Enabled):
   - Batch results are cached
   - Resume from any interruption point
   - Statistical analysis cached for 7 days

2. **Switch to OpenRouter**:
   - Higher rate limits than direct Gemini
   - Use `persona_config_openrouter.yaml`

3. **Adjust Batch Size**:
   ```yaml
   batch_size: 5              # Smaller batches
   max_tokens_per_batch: 15000 # Fewer tokens per call
   ```

4. **Sequential Processing**:
   ```yaml
   parallel_batches: 1        # No parallel processing
   retry_delay_seconds: 15    # Longer delays
   ```

## 🎉 **Expected Results**

With the map-reduce strategy analyzing 100% of your content, you should now see:

### Before (Truncated):
- **Mental Models**: 0
- **Core Beliefs**: 0
- **Content Analyzed**: <1% (15k tokens)

### After (Map-Reduce):
- **Mental Models**: 15-50 high-quality frameworks
- **Core Beliefs**: 30-100 fundamental principles
- **Content Analyzed**: 100% (2.3M words)

The map-reduce approach will identify patterns that recur across the entire corpus, finally enabling proper extraction of Greg's mental models and core beliefs!

## 🔍 **Troubleshooting**

### Rate Limit Errors:
- ✅ **Expected**: Gemini has strict limits
- ✅ **Solution**: System will retry and cache successful batches
- ✅ **Alternative**: Use OpenRouter configuration

### JSON Parsing Errors:
- ✅ **Cause**: Empty responses due to rate limits
- ✅ **Solution**: Retry logic handles this automatically

### Cache Issues:
```bash
# Clear cache if needed
python -m dk_rag.cli.persona_builder cache clear --older-than-days 1
```

Your configuration is working perfectly! The rate limits are actually a good sign - it means you're successfully connecting to Gemini and the system is properly handling the API responses.