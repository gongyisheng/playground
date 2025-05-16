import React, { useEffect, useState, useRef } from "react";
import { ENDPOINT_FILE } from "../constants";

function UserInput({ threadId, onMessageChange, onFilesChange, onSubmit }) {
  const [message, setMessage] = useState("");
  const textAreaRef = useRef(null);
  const [files, setFiles] = useState([]);
  const [filesUploadStatus, setFilesUploadStatus] = useState({});

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(message, files);
    setMessage("");
    setFiles([]);
    setFilesUploadStatus({});
  };

  const handleTextChange = (e) => {
    setMessage(e.target.value);
    onMessageChange(e.target.value);
  };

  const handleFilesAdd = (e) => {
    let newFiles = [...files];
    let toUploadFiles = [];
    if (e.target.files.length + newFiles.length > 5) {
      alert("Cannot upload more than 5 files.");
      return;
    }
    for (let file of e.target.files) {
      if (file.size > 1048576*32) {
        alert("Cannot upload file larger than 32MB: " + file.name);
        continue
      }
      // dedup
      if (newFiles.some((filename) => filename === file.name)) {
        // remove old file
        newFiles = newFiles.filter((filename) => filename !== file.name);
      }
      toUploadFiles.push(file);
      newFiles.push(file.name);
    }
    console.log("newFiles", newFiles);
    setFiles(newFiles);
    onFilesChange(newFiles);
    for (let file of toUploadFiles) {
      handleFileUpload(file);
    }
  }

  const handleFilesRemove = (e) => {
    let newFiles = [...files];
    const fileName = e.target.parentElement.parentElement.children[0].textContent;
    newFiles = newFiles.filter((f) => f.name !== fileName);
    setFiles(newFiles);
    onFilesChange(newFiles);
    setFilesUploadStatus((prev) => {
      const newStatus = { ...prev };
      delete newStatus[fileName];
      return newStatus;
    });
  }

  const handleFileUpload = async (file) => {
    const formData = new FormData();
    formData.append("files", file);

    try {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${ENDPOINT_FILE}?thread_id=${threadId}`, true);
      xhr.withCredentials = true;
    
      xhr.upload.onprogress = (event) => {
        const progress = Math.round((event.loaded / event.total) * 100);
        setFilesUploadStatus((prev) => ({ ...prev, [file.name]: progress }));
      }

      xhr.onload = () => {
        if (xhr.status === 200) {
          setFilesUploadStatus((prev) => ({ ...prev, [file.name]: 100 }));
        } else {
          setFilesUploadStatus((prev) => ({ ...prev, [file.name]: "fail" }));
          alert("Upload file " + file.name + " failed.");
        }
      }

      xhr.send(formData);
    } catch (error) {
      setFilesUploadStatus((prev) => ({ ...prev, [file.name]: "fail" }));
      alert("Upload file " + file.name + " failed.");
    }
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
          {files.map((filename) => (
            <div key={filename} className="flex items-center">
              <div className="px-2">{filename}</div>
              {filesUploadStatus[filename] === "fail" && (
                <div className="text-red-500">Failed</div>
              )}
              {filesUploadStatus[filename] === 100 && (
                <div className="text-green-500">100%</div>
              )}
              {filesUploadStatus[filename] !== 100 && (
                <div className="text-green-500">{filesUploadStatus[filename]}%</div>
              )}
              <button onClick={handleFilesRemove}>
                <img src="./static/close.png" className="max-w-4 max-h-4 ml-2" />
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default UserInput;
