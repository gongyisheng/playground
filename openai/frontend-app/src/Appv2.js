// App.js

import React, { useState } from "react";
import UserInput from "./components/UserInput";
import ChatDisplay from "./components/ChatDisplay";

function App() {
  const [userMessage, setUserMessage] = useState("");
  const [conversations, setConversations] = useState([]);

  const handleUserInputChange = (newContext) => {
    setUserMessage(newContext);
  };

  const appendMessage = (role, message) => {
    setConversations([
      ...conversations,
      {
        role: role,
        message: message,
      },
    ]);
  };

  const handleConversationSubmit = (message) => {
    appendMessage("user", message);
  };

  return (
    <div>
      <div className="container mx-auto p-4">
        <ChatDisplay conversations={conversations} />
      </div>
      <div className="container mx-auto p-4">
        <UserInput
          onContextChange={handleUserInputChange}
          onSubmit={handleConversationSubmit}
        />
      </div>
    </div>
  );
}

export default App;
