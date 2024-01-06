import React, { useEffect, useState } from 'react';

const SSEDisplay = ( uuid ) => {
  const [sseData, setSSEData] = useState([]);

  useEffect(() => {
    const sse = new EventSource('http://localhost:5600?uuid=' + uuid.uuid);
    console.log('uuid:', uuid.uuid, 'Type', typeof uuid.uuid)

    // Event listener for SSE messages
    sse.onmessage = (event) => {
      console.log(event.data);
      setSSEData((prev) => [...prev, event.data]);
    };

    // Event listener for SSE errors
    sse.onerror = (error) => {
      console.error('EventSource failed:', error);
      sse.close();
    };  

    return () => {
      sse.close();
    };
  }, []);

  return (
    <div>
      <h2>SSE Data:</h2>
      <p>{sseData}</p>
    </div>
  );
};

export default SSEDisplay;