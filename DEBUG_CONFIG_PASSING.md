# Config Passing Debug - Next Steps

## Changes Made

### 1. Updated `langchain_client.py`
- Added import: `from langchain_core.runnables import RunnableConfig`
- Changed `call_model` signature from `config: Optional[Dict[str, Any]] = None` to `config: RunnableConfig`
- Added debug logging to see what config is actually received

### 2. Updated `main.py`
- Added debug logging to show config being passed to `ainvoke`

## What to Do Now

### 1. Restart the Server
```bash
cd /Users/ziruiwang/Projects/chatbot-project/agent-server
python main.py
```

### 2. Test Your Query Again

Upload your resume and send a chat message. You should now see additional debug output like:

```
📨 Received stream request - thread_id: e14500bc-782d-4f85-8aad-6f7d9be633d1
✓ Using existing thread config for: e14500bc-782d-4f85-8aad-6f7d9be633d1
💬 Streaming query: 'with uploaded document, can you tell me...'
🔧 Config being passed: {'configurable': {'thread_id': 'e14500bc-...'}}
🔧 Config type: <class 'dict'>
🔧 call_model - config type: <class 'dict'>
🔧 call_model - has configurable: True
🔧 call_model - config keys: dict_keys(['configurable'])
🔧 call_model - thread_id extracted: e14500bc-782d-4f85-8aad-6f7d9be633d1
🔍 Should use RAG: True (thread_id=e14500bc-782d-4f85-8aad-6f7d9be633d1)
```

### 3. Check for Issues

If you still see `thread_id=None`, the debug logs will tell us:
- What type config is (dict, RunnableConfig, etc.)
- What keys it contains
- Why thread_id extraction is failing

### 4. Expected Behavior

**GOOD** - You should see:
```
🔧 call_model - thread_id extracted: e14500bc-...
📚 Thread e14500bc has documents in vector store
🔍 Query: 'with uploaded document...' (should_use_rag=True)
✓ RAG: Retrieved 6 relevant document chunks
```

**BAD** - If you see:
```
🔧 call_model - thread_id extracted: None
🔍 Should use RAG: None (thread_id=None)
```
Then the debug logs will show us why.

## Possible Issues

### Issue 1: Config Type Mismatch
If config is a `RunnableConfig` object instead of dict, we may need to access it differently:
```python
thread_id = config.get("thread_id")  # Instead of config["configurable"]["thread_id"]
```

### Issue 2: LangGraph Not Passing Config
LangGraph might not be passing config to node functions by default. We may need to:
```python
# In create_workflow()
workflow.add_node("model", RunnableLambda(self.call_model).with_config())
```

### Issue 3: Config Lost in Checkpoint
The MemorySaver might be interfering with config passing.

## If Still Broken After Restart

Share the full debug output from the server console, specifically:
1. The `🔧 Config being passed` lines from main.py
2. The `🔧 call_model` lines from langchain_client.py
3. Any errors or warnings

This will tell us exactly where the config is being lost.

## Quick Test Command

```bash
# In terminal 1: Start server
cd /Users/ziruiwang/Projects/chatbot-project/agent-server
python main.py

# In terminal 2: Test with curl
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tell me about my Regeneron experience",
    "thread_id": "e14500bc-782d-4f85-8aad-6f7d9be633d1"
  }'
```

Replace the thread_id with your actual thread ID from the document upload response.
