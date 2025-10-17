# RAG System Debugging Guide

## Problem
After uploading a resume and asking "with uploaded resume, can you tell me the work I did at Regeneron, what are my responsibilities", the system responded: "I apologize, but I cannot access or 'upload' resumes..."

This indicates the RAG (Retrieval Augmented Generation) system is not properly retrieving or using the uploaded document context.

## Changes Made

### 1. Enhanced Document Keyword Detection (`langchain_client.py` line ~543)
**Added resume-specific keywords** to help the system recognize when queries are about personal documents:

```python
document_keywords = [
    # ... existing keywords ...
    # Resume/CV specific keywords (NEW)
    "resume", "cv", "curriculum vitae", "work experience", "employment",
    "responsibilities", "worked at", "position", "job", "role",
    "education", "skills", "experience", "background"
]
```

**Why**: The original system only had general document keywords like "uploaded", "document", "paper". Resume-specific queries might not have triggered RAG retrieval.

### 2. Improved RAG System Prompt (`langchain_client.py` line ~645)
**Changed from generic to context-aware prompt**:

**Before**:
```python
"You are a helpful assistant. You have access to document excerpts..."
```

**After**:
```python
"You are a helpful assistant with access to the user's uploaded documents. 
These documents may include resumes, CVs, reports, articles, or other personal/professional materials..."
```

**Why**: The generic prompt didn't make it clear that the documents are the USER'S personal information. The model might have been too cautious about claiming knowledge of personal details.

### 3. Added Debug Logging (`langchain_client.py` line ~627)
**Added extensive logging** to help diagnose RAG issues:

```python
print(f"📚 Thread {thread_id[:8]} has documents in vector store")
print(f"🔍 Query: '{user_query[:100]}...' (should_use_rag={should_retrieve})")
print(f"✓ RAG: Retrieved {len(retrieved_docs)} relevant document chunks")
print("⚠️ RAG: No relevant documents retrieved despite should_retrieve=True")
print(f"ℹ️ RAG: Skipped retrieval (query not deemed document-related)")
```

**Why**: Without logs, it's impossible to know what's happening internally. These logs will show:
- Whether documents exist in the thread
- Whether RAG was triggered for your query
- How many document chunks were retrieved
- Why RAG might have been skipped

### 4. Increased Retrieval Count (`langchain_client.py` line ~636)
**Changed from k=4 to k=6** document chunks:

```python
retrieved_docs = self.retrieve_context_for_query(thread_id, user_query, k=6)
```

**Why**: Resumes often have multiple sections (experience, education, skills). Retrieving more chunks (6 instead of 4) increases the chance of capturing relevant information about specific companies like Regeneron.

## How to Debug

### Method 1: Use the Debug Script (Recommended)

1. **Make sure your server is running**:
```bash
cd /Users/ziruiwang/Projects/chatbot-project/agent-server
python main.py
```

2. **In a new terminal, run the debug script**:
```bash
cd /Users/ziruiwang/Projects/chatbot-project/agent-server
python test_rag_debug.py
```

3. **Follow the prompts** - it will:
   - Create a new thread
   - Upload your resume
   - Test multiple RAG queries
   - Show whether document context is being used

4. **Watch BOTH terminals**:
   - The test script shows the responses
   - The server console shows the RAG debug logs

### Method 2: Manual Testing with Logging

1. **Restart your server** to load the new code:
```bash
cd /Users/ziruiwang/Projects/chatbot-project/agent-server
python main.py
```

2. **Upload your resume** through your client/UI

3. **Ask your question** about Regeneron

4. **Check the server console** for debug messages:

**What to look for:**

✅ **Good signs** (RAG is working):
```
📚 Thread xxxxxxxx has documents in vector store
🔍 Query: 'with uploaded resume, can you tell me...' (should_use_rag=True)
✓ RAG: Retrieved 6 relevant document chunks
✓ RAG: Augmented prompt with 6 document excerpts
```

⚠️ **Warning signs** (RAG might not work):
```
⚠️ RAG: No relevant documents retrieved despite should_retrieve=True
```
→ This means retrieval was attempted but found no matches. Try simpler keywords.

```
ℹ️ RAG: Skipped retrieval (query not deemed document-related)
```
→ The system thinks your query isn't about documents. The keyword list needs updating.

❌ **Bad signs** (RAG not triggered):
```
(No messages about documents or retrieval at all)
```
→ The thread may not have the document, or thread_id is wrong.

## Common Issues and Solutions

### Issue 1: "No relevant documents retrieved"
**Symptoms**: RAG triggers but finds no matches
**Solution**: 
- The embedding similarity might be too strict
- Try rewording your query with exact words from your resume
- Check if the document was chunked properly (look at num_chunks in upload response)

### Issue 2: "Skipped retrieval"
**Symptoms**: RAG doesn't trigger despite having documents
**Solution**:
- Your query needs more document-related keywords
- Try phrases like "based on my uploaded resume" or "according to my CV"
- Add more keywords to the `document_keywords` list if needed

### Issue 3: Model says "I cannot access documents"
**Symptoms**: RAG works but model refuses to answer
**Solution**:
- This was the ORIGINAL problem - should be fixed by the improved system prompt
- If it persists, the model might be using cached behavior
- Try a fresh thread/session

### Issue 4: Generic answers without document specifics
**Symptoms**: Model answers generally but doesn't cite your resume
**Solution**:
- Increase k (number of chunks) even more (try k=8 or k=10)
- Reduce chunk_size to make chunks more specific (try 800 instead of 1000)
- Ask more specific questions: "What were my specific responsibilities at Regeneron?"

## Testing Different Query Styles

Try these variations to see which works best:

1. **Direct reference**: "According to my uploaded resume, what did I do at Regeneron?"
2. **Specific keywords**: "Tell me about my Regeneron experience"
3. **Document framing**: "In the document I uploaded, what are my responsibilities?"
4. **Section-specific**: "What does my resume say about my work at Regeneron?"

## Monitoring Performance

Watch for these metrics in the logs:
- **Upload time**: Should be < 5 seconds for typical resumes
- **Chunks created**: Should be 5-20 for a typical 1-2 page resume
- **Retrieval time**: Should be < 500ms
- **Response time**: Should be 2-5 seconds total

## Next Steps

1. **Run the debug script** to see exactly what's happening
2. **Check the server logs** for the debug messages added
3. **If RAG is working but answers are wrong**:
   - The model might need better prompting
   - Consider increasing context chunks (k parameter)
   - Check if the resume was properly parsed (check num_chunks)

4. **If RAG is not triggering**:
   - Add more keywords to `document_keywords`
   - Adjust the `should_use_rag` logic
   - Verify document was actually uploaded (check `/documents/list`)

5. **If everything looks right but model won't answer**:
   - The system prompt might need more refinement
   - Try a different model or temperature setting
   - Check if there's a safety filter blocking personal info

## Configuration Tuning

If issues persist, try adjusting these parameters in `langchain_client.py`:

```python
# Line ~371: Chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,        # Smaller chunks = more precise matches
    chunk_overlap=300,     # More overlap = better context continuity
    add_start_index=True
)

# Line ~636: Number of chunks to retrieve
retrieved_docs = self.retrieve_context_for_query(thread_id, user_query, k=8)
```

## Contact/Support

If you're still having issues after trying the above:
1. Share the **full server console output** from the debug script
2. Share a **sanitized version** of your query and response
3. Share the **num_chunks** value from document upload
4. Check if the issue happens with other documents (test with a simple text file)
