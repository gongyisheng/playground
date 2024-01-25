import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

import UserInput from "../components/UserInput";
import ChatDisplay from "../components/ChatDisplay";
import Model from "../components/Model";
import PromptConsole from "../components/PromptConsole";
import { ENDPOINT_AUDIT, ENDPOINT_CHAT, ENDPOINT_PROMPT } from "../constants";

const DEFAULT_PROMPT_CONTENT = "You're a helpful assistant.";

var userMessage = "";
var conversation = [];
var threadId = "";
var model = "GPT-3.5";

function Playground() {
  const navigate = useNavigate();
  const [promptName, setPromptName] = useState("");
  const [promptContent, setPromptContent] = useState("");
  const [promptNote, setPromptNote] = useState("");
  const [refreshFlag, setRefreshFlag] = useState(false);
  const [SSEStatus, setSSEStatus] = useState(false);
  const [SSEData, setSSEData] = useState("");
  const [myPrompts, setMyPrompts] = useState({});
  const [cost, setCost] = useState("$0.00");
  const [limit, setLimit] = useState("$0.00");

  useEffect(
    () => async () => {
      try {
        const response = await fetch(ENDPOINT_PROMPT, {
          headers: {
            "Content-Type": "application/json",
          },
          credentials: "include", // <-- includes cookies in the request
          method: "GET",
        });
        if (response.status === 200) {
          setMyPrompts(await response.json());
        } else if (response.status === 401) {
          // redirect to signin page if not logged in
          navigate("/signin");
          setMyPrompts({});
        }
      } catch (error) {
        setMyPrompts({});
      }
      refrestCostAndLimit();
    },
    
    [refreshFlag],
  );

  const handleModelChange = (_model) => {
    model = _model;
  };

  const handleUserInputChange = (message) => {
    userMessage = message;
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

  const handleRestore = () => {
    setPromptName("");
    setPromptContent("");
    setPromptNote("");
    userMessage = "";
    conversation = [];
    setSSEStatus(false);
    setSSEData("");
    threadId = "";
    setMyPrompts({});
    setRefreshFlag(!refreshFlag);
  };

  const handleSave = async () => {
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
      // redirect to signin page if not logged in
      navigate("/signin");
    }
    else {
      alert("Save prompt failed.");
    }
  };

  const appendToConversation = (role, content) => {
    conversation = conversation.concat({
      role: role,
      content: content,
    });
  };

  const handleSSEMessageUpdate = (token) => {
    setSSEData((prev) => prev + token);
  };

  async function sendChatRequest() {
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
      // redirect to signin page if not logged in
      navigate("/signin");
    } else {
      const data = await response.json();
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
    }
    
    const sse = new EventSource(
      `${ENDPOINT_CHAT}?thread_id=${threadId}&model=${model}`,
      { withCredentials: true }
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

    return () => {
      sse.close();
      refrestCostAndLimit();
    }
  }, [SSEStatus]);

  // Flush SSEData to conversation
  useEffect(() => {
    if (!SSEStatus) {
      if (
        conversation.length > 0 &&
        conversation[conversation.length - 1].role === "user"
      ) {
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
      if (promptContent === "") {
        appendToConversation("system", DEFAULT_PROMPT_CONTENT);
      } else {
        appendToConversation("system", promptContent);
      }
    }
    // append user message to chat display
    appendToConversation("user", userMessage);
    // send chat request to backend
    sendChatRequest();
  };

  const refrestCostAndLimit = async () => {
    const response = await fetch(ENDPOINT_AUDIT, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
      credentials: "include", // <-- includes cookies in the request
    });
    if (response.status === 200) {
      const data = await response.json();
      setCost("$"+data.cost);
      setLimit("$"+data.limit);
    } else if (response.status === 401) {
      // redirect to signin page if not logged in
      navigate("/signin");
    }
  }

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
            onContextChange={handleUserInputChange}
            onSubmit={handleUserMessageSubmit}
          />
        </div>
      </div>
      <div className="col-span-3 overflow-y-scroll">
        <PromptConsole
          myPrompts={myPrompts}
          onPromptNameChange={handlePromptNameChange}
          onPromptContentChange={handlePromptContentChange}
          onPromptNoteChange={handlePromptNoteChange}
          onSave={handleSave}
          onRestore={handleRestore}
        />
      </div>
    </div>
  );
}

export default Playground;
