import React, { useState } from "react";

const modelIcons = {
  "GPT-3.5": "./static/gpt3.5.png",
  "GPT-4": "./static/gpt4.png",
};

function Model({ onChange }) {
  const [chosenModel, setChosenModel] = useState("GPT-3.5");

  function renderButtonStyle(model) {
    var baseStyle = "flex items-center px-2 py-2 w-full m-1";
    if (model === chosenModel) {
      return baseStyle + " rounded-lg border-4 border-double border-black";
    } else {
      return (
        baseStyle +
        " hover:rounded-lg hover:border-4 hover:border-dotted hover:border-black"
      );
    }
  }

  function renderModelBlock(model, onClick) {
    return (
      <button
        className={renderButtonStyle(model)}
        onClick={onClick(model)}
      >
        <img
          src={modelIcons[model]}
          className="max-w-12 max-h-12"
          alt={model}
        />
        <span className="text-base w-full">{model}</span>
      </button>
    );
  }
  const onClick = (model) => (e) => {
    e.preventDefault();
    setChosenModel(model);
    onChange(model);
  };

  return (
    <div className="px-4 py-4 gap-2">
      {renderModelBlock("GPT-3.5", onClick)}
      {renderModelBlock("GPT-4", onClick)}
    </div>
  );
}

export default Model;
