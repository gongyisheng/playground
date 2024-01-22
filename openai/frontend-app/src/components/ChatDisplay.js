// ChatDisplay.js

import React from "react";

const iconSrc = {
  system: "./static/system.png",
  user: "./static/user.png",
  assistant: "./static/chatbot.png",
};

const displayedRole = {
  system: "System",
  user: "You",
  assistant: "ChatGPT",
};

function ChatDisplay({ conversation, SSEData }) {
  function renderConversation(conversation) {
    return (
      <div className="whitespace-break-spaces break-words hyphens-auto">
        {conversation.map((item, index) => (
          <div key={index} className="py-2 pb-4">
            <div className="flex items-center pb-2">
              <img
                src={iconSrc[item.role]}
                className="max-w-12 max-h-12 mr-2 font-bold"
              />
              <span>{displayedRole[item.role]}</span>
            </div>
            <div className="px-14">{item.content}</div>
          </div>
        ))}
      </div>
    );
  }

  function renderSSEData(SSEData) {
    if (SSEData === "") {
      return "";
    } else {
      return (
        <div className="whitespace-break-spaces break-words hyphens-auto">
          <div className="py-2 pb-4">
            <div className="flex items-center pb-2">
              <img
                src={iconSrc["assistant"]}
                className="max-w-12 max-h-12 mr-2 font-bold"
              />
              <span>{displayedRole["assistant"]}</span>
            </div>
            <div className="px-14">{SSEData}</div>
          </div>
        </div>
      );
    }
  }

  return (
    <div>
      {renderConversation(conversation)}
      {renderSSEData(SSEData)}
    </div>
  );
}

export default ChatDisplay;
