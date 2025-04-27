import React, {
  useState,
  FormEvent,
  useRef,
  useEffect,
  useCallback,
  KeyboardEvent,
} from "react";
import "./App.css";

interface ChatMessage {
  sender: "user" | "bot";
  text: string;
}


const MIN_TEXTAREA_HEIGHT = 24; 

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [currentMessage, setCurrentMessage] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const formRef = useRef<HTMLFormElement>(null);

  const adjustTextareaHeight = useCallback(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      const newHeight = Math.max(textarea.scrollHeight, MIN_TEXTAREA_HEIGHT);
      textarea.style.height = `${newHeight}px`;
    }
  }, []);


  useEffect(() => {
    adjustTextareaHeight();
  }, [currentMessage, adjustTextareaHeight]);


  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);


  const performSendMessage = async () => {
    const trimmedMessage = currentMessage.trim();
    if (!trimmedMessage || isLoading) return;

    const newUserMessage: ChatMessage = {
      sender: "user",
      text: trimmedMessage,
    };
    const historyToSend = messages.map((msg) => ({
      role: msg.sender === "user" ? "user" : "assistant", 
      content: msg.text,
    }));
    // historyToSend.push({ role: "user", content: trimmedMessage });
    const MAX_HISTORY_LENGTH = 20;
    const truncatedHistory = historyToSend.slice(-MAX_HISTORY_LENGTH);
    setMessages((prevMessages) => [...prevMessages, newUserMessage]);
    setCurrentMessage("");
    setIsLoading(true);

    setTimeout(() => {
      // if (textareaRef.current) {
      //   textareaRef.current.style.height = `${MIN_TEXTAREA_HEIGHT}px`;
      // }
    }, 0);

    try {
      const response = await fetch("http://localhost:5001/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ history: truncatedHistory, message: newUserMessage }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.error || `HTTP error! status: ${response.status}`
        );
      }

      const data = await response.json();
      const newBotMessage: ChatMessage = {
        sender: "bot",
        text: data.reply,
      };
      setMessages((prevMessages) => [...prevMessages, newBotMessage]);
    } catch (error) {
      console.error("Failed to send message:", error);
      const errorMessage: ChatMessage = {
        sender: "bot",
        text: `Sorry, I encountered an error: ${
          error instanceof Error ? error.message : String(error)
        }`,
      };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };


  const handleFormSubmit = (e: FormEvent) => {
    e.preventDefault();
    performSendMessage();
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setCurrentMessage(e.target.value);
  };


  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      performSendMessage();
    }
  };

  return (
    <div className="chat-page">
      <div className="chat-container">
        <div className="message-list">
          {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.sender}`}>
              <p style={{ whiteSpace: "pre-wrap" }}>{msg.text}</p>
            </div>
          ))}
          {isLoading && (
            <div className="message bot loading">
              <p>...</p>
            </div>
          )}
          <div className={'message-end'} ref={messagesEndRef} />
        </div>

        <div className="input-area">
          <form
            ref={formRef}
            onSubmit={handleFormSubmit}
            className="message-input-form"
          >
            <textarea
              ref={textareaRef}
              rows={1}
              value={currentMessage}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              placeholder="Message..."
              disabled={isLoading}
              className={'input-textarea'}
            />
            <button type="submit" disabled={isLoading || !currentMessage.trim()}>
              <img src="../assets/up-arrow-svgrepo-com%20(1).svg" className="send-icon"/>
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;
