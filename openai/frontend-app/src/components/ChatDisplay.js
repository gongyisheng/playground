// ChatDisplay.js

import React from "react";

function ChatDisplay({ conversations }) {
  return (
    <div className="mb-4">
      {conversations.map((item, index) => (
        <div key={index} className={item.user ? "text-right" : "text-left"}>
          {item.role}: {item.message}
        </div>
      ))}
    </div>
  );
}

export default ChatDisplay;
