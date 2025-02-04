import React, { useState } from "react";

const modelIcons = {
  "GPT4o-mini": "./static/gpt-weak.png",
  "GPT4o": "./static/gpt-strong.png",
  "o3-mini": "./static/o3-mini.png",
  "o1": "./static/o1.png",
  "Deepseek-v3": "./static/deepseek-v3.png",
  "Deepseek-r1": "./static/deepseek-r1.png",
  "Llama3.2": "./static/llama3.png",
  "Claude3.5": "./static/claude.png",
};

const extraText = {}

function Model({ onChange, cost, limit }) {
  const [chosenModel, setChosenModel] = useState("GPT4o-mini");

  function renderButtonStyle(model) {
    var baseStyle = "flex items-center w-full px-2 py-2 m-1";
    if (model === chosenModel) {
      return baseStyle + " rounded-lg border-4 border-double border-black";
    } else {
      return (
        baseStyle +
        " hover:rounded-lg hover:border-4 hover:border-dotted hover:border-black active:border-4 active:border-double"
      );
    }
  }

  function renderModelText(model) {
    if (model in extraText) { 
      return (
        <div className="text-base w-full justify-center">
          <div>{model}</div>
          <div className="text-xs">{extraText[model]}</div>
        </div>
      );
    } else {
      return <div className="text-base w-full justify-center">{model}</div>;
    }
  }

  function renderModelBlock(model, onClick) {
    return (
      <button className={renderButtonStyle(model)} onClick={onClick(model)}>
        <img
          src={modelIcons[model]}
          className="max-w-12 max-h-12"
          alt={model}
        />
        {renderModelText(model)}
      </button>
    );
  }
  const onClick = (model) => (e) => {
    e.preventDefault();
    setChosenModel(model);
    onChange(model);
  };

  return (
    <div>
      <div className="px-4 py-4 gap-2">
        {renderModelBlock("GPT4o-mini", onClick)}
        {renderModelBlock("GPT4o", onClick)}
        {renderModelBlock("Deepseek-v3", onClick)}
        {renderModelBlock("o3-mini", onClick)}
        {renderModelBlock("o1", onClick)}
        {renderModelBlock("Deepseek-r1", onClick)}
        {/* {renderModelBlock("Llama3.2", onClick)} */}
        {renderModelBlock("Claude3.5", onClick)}
      </div>
      <div className="px-4 py-4 justify-center">
        <span className="text-sm italic">Cost this month: ${cost}/${limit}</span>
      </div>
    </div>
  );
}

export default Model;
