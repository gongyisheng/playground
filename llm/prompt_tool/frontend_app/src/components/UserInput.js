import React, { useEffect, useState, useRef } from "react";

function UserInput({ onMessageChange, onFilesChange, onFilesUpload, onSubmit }) {
  const [message, setMessage] = useState("");
  const textAreaRef = useRef(null);
  const [files, setFiles] = useState([]);

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(message, files);
    setMessage("");
  };

  const handleTextChange = (e) => {
    setMessage(e.target.value);
    onMessageChange(e.target.value);
  };

  const handleFilesAdd = (e) => {
    let newFiles = [...files];
    if (e.target.files.length + newFiles.length > 5) {
      alert("Cannot upload more than 5 files.");
      return;
    }
    for (let file of e.target.files) {
      if (file.size > 104857600) {
        alert("Cannot upload file larger than 100MB: " + file.name);
        continue
      }
      // dedup
      if (newFiles.some((f) => f.name === file.name)) {
        // remove old file
        newFiles = newFiles.filter((f) => f.name !== file.name);
      }
      newFiles.push(file);
    }
    setFiles(newFiles);
    onFilesChange(newFiles);
  }

  const handleFilesRemove = (e) => {
    let newFiles = [...files];
    const fileName = e.target.parentElement.parentElement.children[0].textContent;
    newFiles = newFiles.filter((f) => f.name !== fileName);
    setFiles(newFiles);
    onFilesChange(newFiles);
  }

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      handleSubmit(e);
    }
  };

  useEffect(() => {
    textAreaRef.current.style.height = "auto"; // reset the height
    textAreaRef.current.style.height =
      Math.min(textAreaRef.current.scrollHeight, 200) + "px"; // set the height
  }, [message]);

  return (
    <div className="relative items-center rounded-lg border-2 border-black">
      <div className="px-2 pt-2">
        <textarea
          type="text"
          value={message}
          onChange={handleTextChange}
          placeholder="Message ChatBot..."
          className="focus:ring-0 w-full focus:outline-none size-fit resize-none Input-class"
          rows="1"
          onKeyDown={handleKeyDown}
          ref={textAreaRef}
          />
      </div>
      <div className="flex items-center bottom-px px-2">
        <div className="py-1">
        <label className="cursor-pointer">
          <img src="./static/attachment.png" className="max-w-6 max-h-6"/>
          <input type="file" multiple className="hidden" onChange={handleFilesAdd} />
          </label>
        </div>
        <div>
          {files.map((file) => (
            <div key={file.name} className="flex items-center">
              <div className="px-2">{file.name}</div>
              <button onClick={handleFilesRemove}>
                <img src="./static/close.png" className="max-w-4 max-h-4" />
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default UserInput;
