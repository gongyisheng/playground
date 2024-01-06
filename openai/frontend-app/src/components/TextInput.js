// TextInput.js
import React, { useState } from 'react';

const TextInput = ({ onUuidReceived }) => {
  const [inputText, setInputText] = useState('');

  const handleInputChange = (e) => {
    setInputText(e.target.value);
  };

  const handleButtonClick = async () => {
    try {
      const response = await fetch('http://localhost:5600', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText }),
      });

      const uuid = await response.json().then((data) => data.uuid);
      console.log('uuid:', uuid, 'Type', typeof uuid)
      onUuidReceived(uuid);
    } catch (error) {
      console.error('Error fetching UUID:', error);
    }
  };

  return (
    <div>
      <input type="text" value={inputText} onChange={handleInputChange} />
      <button onClick={handleButtonClick}>Get UUID</button>
    </div>
  );
};

export default TextInput;
