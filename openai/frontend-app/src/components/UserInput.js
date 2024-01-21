import React, { useEffect, useState, useRef } from "react";

function UserInput({ onContextChange, onSubmit }) {
  const [input, setInput] = useState("");
  const textAreaRef = useRef(null);

  const handleChange = (e) => {
    setInput(e.target.value);
    onContextChange(e.target.value);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      handleSubmit(e);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(input);
    setInput("");
  };

  useEffect(() => {
    textAreaRef.current.style.height = "auto"; // reset the height
    textAreaRef.current.style.height = Math.min(textAreaRef.current.scrollHeight, 200) + "px"; // set the height
  }, [input]);

  return (
    <form onSubmit={handleSubmit}>
      <div className="flex items-center">
        <textarea
          type="text"
          value={input}
          onChange={handleChange}
          placeholder="Message ChatGPT..."
          className="pl-4 pr-12 py-4 rounded-lg border-2 border-black focus:ring-0 focus:outline-none focus:border-current w-full size-fit resize-none"
          rows="1"
          onKeyDown={handleKeyDown}
          ref={textAreaRef}
        />
        <button
          className="px-2"
          onClick={handleSubmit}
        >
          <img src="./static/up-arrow.png" className="max-w-8 max-h-8" />
        </button>
      </div>
    </form>
  );
}

export default UserInput;
