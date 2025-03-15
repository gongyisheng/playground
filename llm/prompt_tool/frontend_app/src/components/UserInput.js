import React, { useEffect, useState, useRef } from "react";

function UserInput({ onContextChange, onSubmit }) {
  const [input, setInput] = useState("");
  const textAreaRef = useRef(null);
  const [files, setFiles] = useState([])

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(input);
    setInput("");
  };

  const handleTextChange = (e) => {
    setInput(e.target.value);
    onContextChange(e.target.value);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      handleSubmit(e);
    }
  };

  useEffect(() => {
    textAreaRef.current.style.height = "auto"; // reset the height
    textAreaRef.current.style.height =
      Math.min(textAreaRef.current.scrollHeight, 200) + "px"; // set the height
  }, [input]);

  return (
    <div>
      <div className="flex items-center">
        <textarea
          type="text"
          value={input}
          onChange={handleTextChange}
          placeholder="Message ChatBot..."
          className="pl-4 pr-12 py-4 rounded-lg border-2 border-black focus:ring-0 focus:outline-none focus:border-current w-full size-fit resize-none Input-class"
          rows="1"
          onKeyDown={handleKeyDown}
          ref={textAreaRef}
        />
        <label for="uploadFile1" className="px-2 cursor-pointer">
          <img src="./static/attachment.png" className="max-w-10 max-h-10"/>
          <input type="file" id='uploadFile1' class="hidden"/>
        </label>
      </div>
    </div>
  );
}

export default UserInput;
