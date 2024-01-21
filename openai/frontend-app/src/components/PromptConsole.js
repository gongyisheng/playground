import React, { useState } from "react";

const saveIconSrc = "./static/save.png";
const restoreIconSrc = "./static/restore.png";
const promptList = ["a", "b", "c"];

function PromptConsole() {
  // selected prompt name related
  const [selectedPromptName, setSelectedPromptName] = useState("Your prompts");

  const changeButtonTextColor = (fromColor, toColor) => {
    var dropdownButton = document.getElementById("dropdownButton");
    dropdownButton.classList.remove(fromColor);
    dropdownButton.classList.add(toColor);
  };

  const handleSelectedPromptNameChange = (e) => {
    setSelectedPromptName(e.target.id);
    changeButtonTextColor("text-grey-500", "text-black");
    //onPromptChange(e.target.value);
  };

  const renderPromptOptions = () => {
    return promptList.map((promptName) => (
      <li>
        <a
          id={promptName}
          href="#"
          class="block px-4 py-2 hover:bg-gray-100"
          onClick={handleSelectedPromptNameChange}
        >
          {promptName}
        </a>
      </li>
    ));
  };

  // name related
  const [name, setName] = useState("");

  const handleNameChange = (e) => {
    setName(e.target.value);
    //onPromptChange(e.target.value);
  };

  // prompt related
  const [prompt, setPrompt] = useState("");

  const handlePromptChange = (e) => {
    setPrompt(e.target.value);
    //onPromptChange(e.target.value);
  };

  // note related
  const [note, setNote] = useState("");

  const handleNoteChange = (e) => {
    setNote(e.target.value);
    //onPromptChange(e.target.value);
  };

  // actions, save and restore
  const handleSave = () => {
    // TODO
  };

  const handleRestore = () => {
    setSelectedPromptName("Your prompts");
    changeButtonTextColor("text-black", "text-gray-500");
    setName("");
    setPrompt("");
    setNote("");
  };

  const toggleDropdown = () => {
    var dropdown = document.getElementById("dropdown");
    dropdown.classList.toggle("hidden");
  };

  return (
    <div className="px-4 py-4">
      <div className="py-2">
        <button
          id="dropdownButton"
          data-dropdown-toggle="dropdown"
          class="px-4 py-2 rounded-lg border-2 border-black text-gray-500 text-center flex items-center focus:outline-none focus:ring-0 w-full"
          type="button"
          onClick={toggleDropdown}
        >
          {selectedPromptName}
          <svg
            class="w-2.5 h-2.5 ms-3"
            aria-hidden="true"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 10 6"
          >
            <path
              stroke="currentColor"
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="m1 1 4 4 4-4"
            />
          </svg>
        </button>
        <div id="dropdown" className="py-1 hidden">
          <div class="divide-y divide-black rounded-lg border-2 border-black border-dashed w-full">
            <ul
              class="py-2 text-sm text-black"
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
        value={name}
        onChange={handleNameChange}
        placeholder="default-prompt"
        className="pl-4 pr-12 py-2 rounded-lg border-2 border-black focus:ring-0 focus:outline-none focus:border-current w-full"
      />
      <div className="py-2">Prompt</div>
      <textarea
        type="text"
        value={prompt}
        onChange={handlePromptChange}
        placeholder="You're a helpful assistant."
        className="pl-4 pr-12 py-4 rounded-lg border-2 border-black focus:ring-0 focus:outline-none focus:border-current w-full resize-none"
        rows="6"
      />
      <div className="py-2">Note</div>
      <textarea
        type="text"
        value={note}
        onChange={handleNoteChange}
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
