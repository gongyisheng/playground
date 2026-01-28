import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import md5 from "md5";

import { ENDPOINT_SIGNIN, ENDPOINT_AUTH } from "../constants";

const logoSrc = "./logo.png";

const SignIn = () => {
  const navigate = useNavigate();
  const [sessionState, setSessionState] = useState(0);
  const [checkSessionStateEvent, setCheckSessionStateEvent] = useState(0);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [errorMessage, setErrorMessage] = useState("");

  const handleUsernameChange = (e) => {
    setUsername(e.target.value);
  };

  const handlePasswordChange = (e) => {
    setPassword(e.target.value);
  };

  async function handleSignIn(e) {
    e.preventDefault();
    if (username === "") {
      setErrorMessage("Username is not provided.");
      return;
    } else if (password === "") {
      setErrorMessage("Password is not provided.");
      return;
    }

    try {
      const response = await fetch(ENDPOINT_SIGNIN, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        credentials: "include", // <-- includes cookies in the request
        body: JSON.stringify({
          username: username,
          password: md5(password),
        }),
      });
      if (response.status === 200) {
        navigate("/playground");
      } else {
        const data = await response.json();
        setErrorMessage(data.error);
      }
    } catch (err) {
      console.log(err);
      setErrorMessage("Something went wrong. Please try again later.");
    }
  }

  const handleSessionStateChange = (state) => {
    setSessionState(state);
    setCheckSessionStateEvent(checkSessionStateEvent + 1);
  };

  const checkSessionState = async () => {
    try {
      const response = await fetch(ENDPOINT_AUTH, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include", // <-- includes cookies in the request
      });
      if (response.status === 200) {
        handleSessionStateChange(1);
      } else if (response.status === 401) {
        handleSessionStateChange(-1);
      }
    } catch (error) {
      console.error("Failed to check session state:", error);
    }
  }

  useEffect(() => {
    if (sessionState > 0) {
      navigate("/playground");
    } else if (sessionState < 0) {
      navigate("/signin");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [checkSessionStateEvent]);

  useEffect(() => {
    checkSessionState();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="flex min-h-full flex-col justify-center px-6 py-12 lg:px-8">
      <div className="sm:mx-auto sm:w-full sm:max-w-sm">
        <img className="mx-auto h-20 w-auto" src={logoSrc} alt="Your Logo" />
        <h2 className="mt-10 text-center text-2xl font-bold leading-9 tracking-tight">
          Sign in to continue
        </h2>
      </div>

      <div className="mt-10 sm:mx-auto sm:w-full sm:max-w-sm">
        <form className="space-y-6" action="#" onSubmit={handleSignIn}>
          <div>
            <label
              htmlFor="username"
              className="block text-sm font-medium leading-6 text-balance"
            >
              Username
            </label>
            <div className="mt-2">
              <input
                id="username"
                name="username"
                type="username"
                autoComplete="username"
                onChange={handleUsernameChange}
                className="block px-4 py-2 w-full rounded-md text-black border border-black focus:ring-0 focus:outline-none focus:border-2 sm:text-sm sm:leading-6 Input-class"
              />
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between">
              <label
                htmlFor="password"
                className="block text-sm font-medium leading-6 text-black"
              >
                Password
              </label>
            </div>
            <div className="mt-2">
              <input
                id="password"
                name="password"
                type="password"
                autoComplete="current-password"
                onChange={handlePasswordChange}
                className="block px-4 py-2 w-full rounded-md text-black border border-black focus:ring-0 focus:outline-none focus:border-2 focus:border-black sm:text-sm sm:leading-6 Input-class"
              />
            </div>
          </div>

          <div>
            <div>
              <p className="text-xs text-red-500 italic">{errorMessage}</p>
            </div>
            <button
              type="submit"
              className="flex mt-2 w-full justify-center rounded-md bg-red-500 px-4 py-2 text-sm font-semibold leading-6 text-white shadow-sm hover:bg-red-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-red-500"
            >
              Sign in
            </button>
          </div>
        </form>

        <p className="mt-5 text-center text-sm text-gray-400">
          <p className="mt-1">Not a member?{" "}
            <a href="/signup" className="font-semibold leading-6 text-red-500">
              Sign up with invitation code
            </a>
          </p>
          <p className="mt-1">
            It's okay to{" "}
            <a href="/forget_password" className="font-semibold leading-6 text-red-300">
              forget your password
            </a>
          </p>
        </p>
      </div>
    </div>
  );
};

export default SignIn;
