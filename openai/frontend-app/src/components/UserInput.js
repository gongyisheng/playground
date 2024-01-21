import React, { useState } from "react";

function UserInput({ onContextChange, onSubmit }) {
  const [input, setInput] = useState("");

  const handleChange = (e) => {
    setInput(e.target.value);
    onContextChange(e.target.value);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(input);
    setInput("");
  };

  return (
    <form onSubmit={handleSubmit}>
      <div className="flex items-center">
        <input
          type="text"
          value={input}
          onChange={handleChange}
          placeholder="Message ChatGPT..."
          className="pl-4 pr-12 py-2 rounded border-solid focus:outline-none focus:border-transparent w-full"
        />
        <img
          src="./static/up-arrow.png"
          className="max-w-8 max-h-8 ml-4 rounded-md"
          onClick={handleSubmit}
        ></img>
      </div>
    </form>
  );
}

export default UserInput;
