// App.js
import React, { useState, useEffect, forceUpdate } from "react";
import UserInput from "./components/UserInput";
import ChatDisplay from "./components/ChatDisplay";

const BASE_URL = "http://127.0.0.1:5600";
const ENDPOINT_CHAT = BASE_URL + "/chat";
const ENDPOINT_PROMPT = BASE_URL + "/prompt";

var systemMessage = "You're a helpful assistant.";
var userMessage = "";
var conversation = [];
var threadId = "";
const models = ["gpt-3.5", "gpt-4"];
var model = "gpt-3.5-turbo-1106";

function App() {
  const [SSEStatus, setSSEStatus] = useState(false);
  const [SSEData, setSSEData] = useState("");

  const handleUserInputChange = (message) => {
    userMessage = message;
  };

  const appendToConversation = (role, content) => {
    conversation = conversation.concat({
      role: role,
      content: content,
    });
    console.log("append to conversation:", conversation);
  };

  const handleSSEMessageUpdate = (token) => {
    console.log("get sse token:", token);
    setSSEData((prev) => prev + token);
  };

  async function sendChatRequest() {
    console.log("send chat request:", conversation);
    const response = await fetch(ENDPOINT_CHAT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        conversation: conversation,
        thread_id: threadId,
      }),
    });
    const status = response.status;
    const data = await response.json();
    if (status === 200) {
      threadId = data.thread_id;
      setSSEStatus(true);
    } else {
      console.error(
        "error sending chat request:",
        data,
        "conversation:",
        conversation,
      );
    }
  }

  // Get SSE data from backend
  useEffect(() => {
    if (!SSEStatus || conversation.length === 0) {
      return;
    } else {
      console.log("SSEStatus:", SSEStatus);
    }
    const sse = new EventSource(
      `${ENDPOINT_CHAT}?thread_id=${threadId}&model=${model}`,
    );

    // Event listener for SSE messages
    sse.onmessage = (event) => {
      let token = JSON.parse(event.data).content;
      handleSSEMessageUpdate(token);
    };

    sse.onerror = (event) => {
      sse.close();
      setSSEStatus(false);
    };

    return () => sse.close();
  }, [SSEStatus]);

  // Flush SSEData to conversation
  useEffect(() => {
    if (!SSEStatus) {
      if (conversation.length > 0 && conversation[conversation.length - 1].role === "user") {
        console.log("Flush SSEData to conversation");
        appendToConversation("assistant", SSEData);
        setSSEData("");
      }
    }
  }, [SSEStatus]);

  const handleUserMessageSubmit = (userMessage) => {
    // skip empty message
    if (userMessage === "") return;
    // appened system message to chat display
    if (conversation.length === 0) {
      appendToConversation("system", systemMessage);
    }
    // append user message to chat display
    appendToConversation("user", userMessage);
    // send chat request to backend
    sendChatRequest();
  };

  return (
    <div className="px-8">
      <div>
        <ChatDisplay conversation={conversation} SSEData={SSEData} />
      </div>
      <div>
        <UserInput
          onContextChange={handleUserInputChange}
          onSubmit={handleUserMessageSubmit}
        />
      </div>
    </div>
  );
}

export default App;
