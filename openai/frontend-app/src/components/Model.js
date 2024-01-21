const modelIcons = {
  "GPT-3.5": "./static/gpt3.5.png",
  "GPT-4": "./static/gpt4.png",
};

function renderModelBlock(model, onClick) {
  return (
    <button className="flex items-center py-2" onClick={onClick(model)}>
      <img src={modelIcons[model]} className="max-w-12 max-h-12" alt={model} />
      <span className="pl-2 text-base">{model}</span>
    </button>
  );
}

function Model({ onChange }) {
  const onClick = (model) => (e) => {
    e.preventDefault();
    onChange(model);
  };

  return (
    <div className="px-4 py-4">
      {renderModelBlock("GPT-3.5", onClick)}
      {renderModelBlock("GPT-4", onClick)}
    </div>
  );
}

export default Model;
