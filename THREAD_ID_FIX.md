# CRITICAL: Thread ID Issue - Quick Fix Guide

## The Problem

You're seeing `thread_id=None` in the logs, which means:
- The thread ID is not being sent in the chat request
- OR you're using a different thread ID than the one where you uploaded the document

## Why This Happens

When you:
1. Upload a document → Document is saved to `thread_id_A`
2. Send a chat message WITHOUT thread_id → Server creates `thread_id_B` (new thread)
3. Result: You're asking about a document in thread B, but the document is in thread A

## The Fix

**You MUST use the SAME thread_id for both uploading and chatting!**

### Step-by-Step Solution

#### Option 1: Let the Server Create the Thread (Recommended)

1. **Create a new thread first:**
   ```bash
   curl -X POST http://localhost:8000/threads/new
   ```
   
   Response:
   ```json
   {
     "thread_id": "12345678-1234-1234-1234-123456789abc",
     "message": "New thread created"
   }
   ```

2. **Save the thread_id!** You'll need it for all subsequent requests.

3. **Upload your document with this thread_id:**
   ```bash
   curl -X POST "http://localhost:8000/documents/upload?thread_id=12345678-1234-1234-1234-123456789abc" \
     -F "file=@/path/to/your/resume.pdf"
   ```

4. **Chat using the SAME thread_id:**
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Tell me about my work at Regeneron",
       "thread_id": "12345678-1234-1234-1234-123456789abc"
     }'
   ```

#### Option 2: Use the Debug Script

The debug script handles thread management for you:

```bash
cd /Users/ziruiwang/Projects/chatbot-project/agent-server
python test_rag_debug.py
```

This will:
- Create a thread automatically
- Upload your document
- Test queries
- Use the correct thread_id throughout

#### Option 3: Check Your Existing Thread

If you already uploaded a document, find which thread it's in:

1. **Check the upload response** - it should have contained the thread_id
2. **Use the check_thread_status.py script:**
   ```bash
   python check_thread_status.py YOUR_THREAD_ID
   ```

## How to Fix Your Client Code

If you're using a frontend/client, make sure it:

1. **Stores the thread_id** after creating a thread or uploading a document
2. **Sends the thread_id with EVERY request**

### Example in JavaScript/TypeScript:

```javascript
// Store thread_id in state/localStorage
let threadId = null;

// When uploading document
async function uploadDocument(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  // Create thread first if needed
  if (!threadId) {
    const threadResp = await fetch('http://localhost:8000/threads/new', {
      method: 'POST'
    });
    const threadData = await threadResp.json();
    threadId = threadData.thread_id; // SAVE THIS!
  }
  
  // Upload with thread_id
  const response = await fetch(
    `http://localhost:8000/documents/upload?thread_id=${threadId}`,
    {
      method: 'POST',
      body: formData
    }
  );
  
  return response.json();
}

// When sending chat message
async function sendMessage(message) {
  const response = await fetch('http://localhost:8000/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: message,
      thread_id: threadId  // USE THE SAME THREAD_ID!
    })
  });
  
  return response.json();
}
```

### Example in Python:

```python
import requests

class ChatClient:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.thread_id = None
    
    def create_thread(self):
        """Create a new thread and store the ID."""
        response = requests.post(f"{self.base_url}/threads/new")
        data = response.json()
        self.thread_id = data['thread_id']
        print(f"Created thread: {self.thread_id}")
        return self.thread_id
    
    def upload_document(self, file_path):
        """Upload a document to the current thread."""
        if not self.thread_id:
            self.create_thread()
        
        with open(file_path, 'rb') as f:
            files = {'file': f}
            params = {'thread_id': self.thread_id}
            response = requests.post(
                f"{self.base_url}/documents/upload",
                files=files,
                params=params
            )
        
        return response.json()
    
    def chat(self, message):
        """Send a chat message."""
        if not self.thread_id:
            self.create_thread()
        
        response = requests.post(
            f"{self.base_url}/chat",
            json={
                'query': message,
                'thread_id': self.thread_id  # Always include thread_id!
            }
        )
        
        return response.json()

# Usage
client = ChatClient()
client.upload_document("resume.pdf")
response = client.chat("Tell me about my work at Regeneron")
print(response['response'])
```

## Verification Steps

After making changes, verify it's working:

1. **Check server logs** - you should see:
   ```
   📨 Received chat request - thread_id: 12345678-...
   📚 Thread has 1 document(s) uploaded
   🔍 Should use RAG: True (thread_id=12345678-...)
   ✓ RAG: Retrieved 6 relevant document chunks
   ```

2. **NOT this:**
   ```
   🔍 Should use RAG: None (thread_id=None)  ← BAD!
   ```

## Quick Test

Run this to test your setup:

```bash
# Terminal 1: Start server
cd /Users/ziruiwang/Projects/chatbot-project/agent-server
python main.py

# Terminal 2: Run test
cd /Users/ziruiwang/Projects/chatbot-project/agent-server
python test_rag_debug.py
```

Watch Terminal 1 (server) for the debug messages!

## API Changes Made

The API now supports BOTH field names:
- `query` (preferred)
- `message` (for backward compatibility)

So these are equivalent:
```json
{"query": "Hello", "thread_id": "123"}
{"message": "Hello", "thread_id": "123"}
```

## Still Not Working?

If you still see `thread_id=None`:

1. **Print the request** in your client code to verify thread_id is being sent
2. **Check the server logs** for the "📨 Received chat request" line
3. **Use check_thread_status.py** to verify documents are in the thread
4. **Try the test_rag_debug.py script** to confirm the server works

## Contact Info

If you need help, provide:
1. The thread_id from document upload response
2. The thread_id you're sending in chat request
3. Full server log output from the chat request
