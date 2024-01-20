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
    <form onSubmit={handleSubmit} className="mb-4">
      <input
        type="text"
        value={input}
        onChange={handleChange}
        placeholder="Type your context here"
        className="border p-2 w-full"
      />
      <button type="submit" className="mt-2 bg-blue-500 text-white p-2">
        Submit
      </button>
    </form>
  );
}

export default UserInput;
