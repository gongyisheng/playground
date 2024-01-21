import React, { useState } from "react";

const saveIconSrc = "./static/save.png";
const restoreIconSrc = "./static/restore.png";

function PromptConsole() {
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

    // description related
    const [description, setDescription] = useState("");

    const handleDescriptionChange = (e) => {
        setDescription(e.target.value);
        //onPromptChange(e.target.value);
    };

    // actions, save and restore
    const handleSave = () => {
        // TODO
    }

    const handleRestore = () => {
        setName("");
        setPrompt("");
        setDescription("");
    }

    return (
        <div className="px-4 py-4">
            <div className="py-2">Select Prompt</div>
            <div className="py-2">Name</div>
            <input
                type="text"
                value={name}
                onChange={handleNameChange}
                placeholder="default-prompt"
                className="pl-4 pr-12 py-4 rounded-lg border-2 border-black focus:ring-0 focus:outline-none focus:border-current w-full"
            />
            <div className="py-2">Prompt</div>
            <textarea
                type="text"
                value={prompt}
                onChange={handlePromptChange}
                placeholder="You're a helpful assistant."
                className="pl-4 pr-12 py-4 rounded-lg border-2 border-black focus:ring-0 focus:outline-none focus:border-current w-full resize-none"
                rows="7"
                />
            <div className="py-2">Description</div>
            <textarea
                type="text"
                value={description}
                onChange={handleDescriptionChange}
                placeholder="Default prompt of ChatGPT."
                className="pl-4 pr-12 py-4 rounded-lg border-2 border-black focus:ring-0 focus:outline-none focus:border-current w-full resize-none"
                rows="7"
            />
            <div className="grid grid-cols-2 justify-items-center py-2">
            <button className="flex items-center grow px-1 py-1 hover:rounded-lg hover:border-2 hover:border-dashed hover:border-black active:border-4 active:border-double" onClick={handleSave}>
                <img src={saveIconSrc} className="max-w-8 max-h-8" />
                <span className="px-2 text-lg">save</span>
            </button>
            <button className="flex items-center grow px-1 py-1 hover:rounded-lg hover:border-2 hover:border-dashed hover:border-black active:border-4 active:border-double" onClick={handleRestore}>
                <img src={restoreIconSrc} className="max-w-8 max-h-8" />
                <span className="px-2 text-lg">restore</span>
                </button>
            </div>
        </div>
    )
}
export default PromptConsole