import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { v4 as uuidv4 } from 'uuid';

import UserInput from "../components/UserInput";
import ChatDisplay from "../components/ChatDisplay";
import Model from "../components/Model";
import PromptConsole from "../components/PromptConsole";
import { ENDPOINT_AUTH, ENDPOINT_AUDIT, ENDPOINT_CHAT, ENDPOINT_PROMPT } from "../constants";

const DEFAULT_PROMPT_CONTENT = "You're a helpful assistant.";

var userMessage = "";
var userFiles = [];
var conversation = [];
var threadId = uuidv4();
var model = "GPT4.1-mini";

function Playground() {
  const navigate = useNavigate();
  const [promptName, setPromptName] = useState("");
  const [promptContent, setPromptContent] = useState("");
  const [promptNote, setPromptNote] = useState("");
  const [refreshFlag, setRefreshFlag] = useState(false);
  const [SSEStatus, setSSEStatus] = useState(false);
  const [SSEData, setSSEData] = useState("");
  const [myPrompts, setMyPrompts] = useState({});
  const [cost, setCost] = useState(0.0);
  const [limit, setLimit] = useState(0.0);
  const [sessionState, setSessionState] = useState(0);
  const [checkSessionStateEvent, setCheckSessionStateEvent] = useState(0);

  const handleModelChange = (_model) => {
    model = _model;
  };

  const handleUserMessageChange = (message) => {
    userMessage = message;
  };

  const handleUserFilesChange = (files) => {
    userFiles = files;
  };

  const handlePromptNameChange = (name) => {
    setPromptName(name);
  };

  const handlePromptContentChange = (content) => {
    setPromptContent(content);
  };

  const handlePromptNoteChange = (note) => {
    setPromptNote(note);
  };

  const handleSessionStateChange = (state) => {
    setSessionState(state);
    setCheckSessionStateEvent(checkSessionStateEvent + 1);
  };

  const handleRestore = () => {
    setPromptName("");
    setPromptContent("");
    setPromptNote("");
    userMessage = "";
    conversation = [];
    setSSEStatus(false);
    setSSEData("");
    threadId = uuidv4();
    setMyPrompts({});
    setRefreshFlag(!refreshFlag);
  };

  const handleSavePrompt = async () => {
    try {
      const response = await fetch(ENDPOINT_PROMPT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include", // <-- includes cookies in the request
        body: JSON.stringify({
          promptName: promptName,
          promptContent: promptContent,
          promptNote: promptNote,
        }),
      });
      if (response.status === 200) {
        alert("Save prompt success.");
      } else if (response.status === 401) {
        handleSessionStateChange(-1);
      }
      else {
        alert("Save prompt failed.");
      }
    } catch (error) {
      console.error("Failed to save prompt:", error);
    }
  };

  const appendToConversation = (role, text, files = []) => {
    var item = {
      role: role,
      content: text
    }
    if (files.length > 0) {
      item.files = files;
    }
    conversation = conversation.concat(item);   
  };

  const handleSSEMessageUpdate = (token) => {
    setSSEData((prev) => prev + token);
  };

  async function sendChatRequest() {
    try {
      const response = await fetch(ENDPOINT_CHAT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include", // <-- includes cookies in the request
        body: JSON.stringify({
          conversation: conversation,
          thread_id: threadId,
        }),
      });
    
      if (response.status === 200) {
        const data = await response.json();
        threadId = data.thread_id;
        setSSEStatus(true);
      } else if (response.status === 401) {
        handleSessionStateChange(-1);
      } else {
        const data = await response.json();
        console.error(
          "error sending chat request:",
          data,
          "conversation:",
          conversation,
        );
      }
    } catch (error) {
      console.error("Failed to send chat request:", error);
    }
  }

  // Get SSE data from backend
  useEffect(() => {
    if (!SSEStatus || conversation.length === 0) {
      return;
    }
    
    const sse = new EventSource(
      `${ENDPOINT_CHAT}?thread_id=${threadId}&model=${model}`,
      {
        withCredentials: true,
        heartbeatTimeout: 120000
      }
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

    return async () => {
      sse.close();
      await refrestCostAndLimit();
    }
  }, [SSEStatus]);

  // Flush SSEData to conversation
  useEffect(() => {
    if (!SSEStatus) {
      if (
        conversation.length > 0 &&
        conversation[conversation.length - 1].role === "user"
      ) {
        appendToConversation("assistant", SSEData, []);
        setSSEData("");
      }
    }
  }, [SSEStatus]);

  const handleUserSubmit = (userMessage) => {
    // skip empty message
    if (userMessage === "") return;
    // skip if files are not uploaded
    // appened system message to chat display
    if (conversation.length === 0) {
      if (promptContent === "") {
        appendToConversation("system", DEFAULT_PROMPT_CONTENT, []);
      } else {
        appendToConversation("system", promptContent, []);
      }
    }
    // append user message to chat display
    if (SSEStatus === false) {
      appendToConversation("user", userMessage, userFiles);
    }
    // send chat request to backend
    sendChatRequest();
    handleUserFilesChange([]);
  };

  const refrestCostAndLimit = async () => {
    try {
      const response = await fetch(ENDPOINT_AUDIT, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include", // <-- includes cookies in the request
      });
      if (response.status === 200) {
        const data = await response.json();
        setCost(data.cost);
        setLimit(data.limit);
      } else if (response.status === 401) {
        handleSessionStateChange(-1);
      }
    } catch (error) {
      console.error("Failed to get audit info:", error);
    }
  }

  const refreshPrompt = async () => {
    try {
      const response = await fetch(ENDPOINT_PROMPT, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include", // <-- includes cookies in the request
      });
      if (response.status === 200) {
        setMyPrompts(await response.json());
      } else if (response.status === 401) {
        setMyPrompts({});
        handleSessionStateChange(-1);
      }
    } catch (error) {
      console.error("Failed to get prompt info:", error);
    }
  };

  const checkSessionState = async () => {
    try {
      const response = await fetch(ENDPOINT_AUTH, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include", // <-- includes cookies in the request
      });
      if (response.status === 200) {
        handleSessionStateChange(1);
      } else if (response.status === 401) {
        handleSessionStateChange(-1);
      }
    } catch (error) {
      console.error("Failed to check session state:", error);
    }
  }

  useEffect(() => {
      refrestCostAndLimit();
      refreshPrompt();
    },
    [refreshFlag],
  );

  useEffect(() => {
    if (sessionState > 0) {
      refrestCostAndLimit();
      refreshPrompt();
    } else if (sessionState < 0) {
      navigate("/signin");
    }
  }, [checkSessionStateEvent]);

  useEffect(() => {
    checkSessionState();
  }, []);

  return (
    <div className="grid grid-cols-12 h-screen max-h-screen">
      <div className="col-span-2">
        <Model onChange={handleModelChange} cost={ cost } limit={ limit } />
      </div>
      <div className="col-span-7 flex flex-col px-8 overflow-y-scroll">
        <div className="grow pt-4">
          <ChatDisplay conversation={conversation} SSEData={SSEData} />
        </div>
        <div className="pb-4">
          <UserInput
            threadId={threadId}
            onMessageChange={handleUserMessageChange}
            onFilesChange={handleUserFilesChange}
            onSubmit={handleUserSubmit}
          />
        </div>
      </div>
      <div className="col-span-3 overflow-y-scroll">
        <PromptConsole
          myPrompts={myPrompts}
          onPromptNameChange={handlePromptNameChange}
          onPromptContentChange={handlePromptContentChange}
          onPromptNoteChange={handlePromptNoteChange}
          onSave={handleSavePrompt}
          onRestore={handleRestore}
        />
      </div>
    </div>
  );
}

export default Playground;
