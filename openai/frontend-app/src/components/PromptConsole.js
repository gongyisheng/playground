import React, { useState } from "react";

const saveIconSrc = "./static/save.png";
const restoreIconSrc = "./static/restore.png";
const promptData = {
  a: {
    name: "a",
    prompt: "You're a helpful assistant. - version a",
    note: "Default prompt of ChatGPT. - version a",
  },
  b: {
    name: "b",
    prompt: "You're a helpful assistant. - version b",
    note: "Default prompt of ChatGPT. - version b",
  },
  c: {
    name: "c",
    prompt: "You're a helpful assistant. - version c",
    note: "Default prompt of ChatGPT. - version c",
  },
};

function PromptConsole({
  onPromptNameChange,
  onPromptContentChange,
  onPromptNoteChange,
  onRestore,
}) {
  // selected prompt name related
  const [selectedPromptName, setSelectedPromptName] = useState("Your prompts");

  const changeButtonTextColor = (fromColor, toColor) => {
    var dropdownButton = document.getElementById("dropdownButton");
    dropdownButton.classList.remove(fromColor);
    dropdownButton.classList.add(toColor);
  };

  const handlePromptOptionClick = (e) => {
    let name = e.target.id;
    setSelectedPromptName(name);
    changeButtonTextColor("text-grey-500", "text-black");
    toggleDropdownForceHidden();
    setPromptName(name);
    setPromptContent(promptData[name].prompt);
    setPromptNote(promptData[name].note);
    //onPromptChange(e.target.value);
  };

  const renderPromptOptions = () => {
    var entries = Object.entries(promptData);
    return entries.map(([k, v]) => (
      <li key={k}>
        <a
          id={v.name}
          href="#"
          className="block px-4 py-2 hover:bg-gray-100"
          onClick={handlePromptOptionClick}
        >
          {v.name}
        </a>
      </li>
    ));
  };

  // prompt name related
  const [promptName, setPromptName] = useState("");

  const handlePromptNameChange = (e) => {
    setPromptName(e.target.value);
    onPromptNameChange(e.target.value);
  };

  // prompt content related
  const [promptContent, setPromptContent] = useState("");

  const handlePromptContentChange = (e) => {
    setPromptContent(e.target.value);
    onPromptContentChange(e.target.value);
  };

  // note related
  const [promptNote, setPromptNote] = useState("");

  const handlePromptNoteChange = (e) => {
    setPromptNote(e.target.value);
    onPromptNoteChange(e.target.value);
  };

  // actions, save and restore
  const handleSave = () => {
    // TODO
  };

  const handleRestore = () => {
    setSelectedPromptName("Your prompts");
    toggleDropdownForceHidden();
    changeButtonTextColor("text-black", "text-gray-500");

    setPromptName("");
    setPromptContent("");
    setPromptNote("");
    onRestore();
  };

  const toggleDropdown = () => {
    var dropdown = document.getElementById("dropdown");
    dropdown.classList.toggle("hidden");
  };

  const toggleDropdownForceHidden = () => {
    var dropdown = document.getElementById("dropdown");
    if (!dropdown.classList.contains("hidden")) {
      dropdown.classList.add("hidden");
    }
  };

  return (
    <div className="px-4 py-4">
      <div className="py-2">
        <button
          id="dropdownButton"
          data-dropdown-toggle="dropdown"
          className="px-4 py-2 rounded-lg border-2 border-black text-gray-500 text-center flex items-center focus:outline-none focus:ring-0 w-full"
          type="button"
          onClick={toggleDropdown}
        >
          {selectedPromptName}
          <svg
            className="w-2.5 h-2.5 ms-3"
            aria-hidden="true"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 10 6"
          >
            <path
              stroke="currentColor"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="m1 1 4 4 4-4"
            />
          </svg>
        </button>
        <div id="dropdown" className="py-1 hidden">
          <div className="divide-y divide-black rounded-lg border-2 border-black border-dashed w-full">
            <ul
              className="py-2 text-sm text-black"
              aria-labelledby="dropdownButton"
            >
              {renderPromptOptions()}
            </ul>
          </div>
        </div>
      </div>
      <div className="py-2">Name</div>
      <input
        type="text"
        value={promptName}
        onChange={handlePromptNameChange}
        placeholder="default-prompt"
        className="pl-4 pr-12 py-2 rounded-lg border-2 border-black focus:ring-0 focus:outline-none focus:border-current w-full"
      />
      <div className="py-2">Prompt</div>
      <textarea
        type="text"
        value={promptContent}
        onChange={handlePromptContentChange}
        placeholder="You're a helpful assistant."
        className="pl-4 pr-12 py-4 rounded-lg border-2 border-black focus:ring-0 focus:outline-none focus:border-current w-full resize-none"
        rows="6"
      />
      <div className="py-2">Note</div>
      <textarea
        type="text"
        value={promptNote}
        onChange={handlePromptNoteChange}
        placeholder="Default prompt of ChatGPT."
        className="pl-4 pr-12 py-4 rounded-lg border-2 border-black focus:ring-0 focus:outline-none focus:border-current w-full resize-none"
        rows="7"
      />
      <div className="grid grid-cols-2 justify-items-center py-2">
        <button
          className="flex items-center grow px-1 py-1 hover:rounded-lg hover:border-2 hover:border-dashed hover:border-black active:border-4 active:border-double"
          onClick={handleSave}
        >
          <img src={saveIconSrc} className="max-w-8 max-h-8" />
          <span className="px-2 text-lg">save</span>
        </button>
        <button
          className="flex items-center grow px-1 py-1 hover:rounded-lg hover:border-2 hover:border-dashed hover:border-black active:border-4 active:border-double"
          onClick={handleRestore}
        >
          <img src={restoreIconSrc} className="max-w-8 max-h-8" />
          <span className="px-2 text-lg">restore</span>
        </button>
      </div>
    </div>
  );
}
export default PromptConsole;
