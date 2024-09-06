import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import md5 from "md5";

import { ENDPOINT_SIGNUP } from "../constants";

const logoSrc = "./logo.png";

const SignUp = () => {
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [invitationCode, setInvitationCode] = useState("");
  const [errorMessage, setErrorMessage] = useState("");

  const handleUsernameChange = (e) => {
    setUsername(e.target.value);
  };

  const handlePasswordChange = (e) => {
    setPassword(e.target.value);
  };

  const handleInvitationCodeChange = (e) => {
    setInvitationCode(e.target.value);
  };

  async function handleSignUp(e) {
    e.preventDefault();
    if (username === "") {
      setErrorMessage("Username is not provided.");
      return;
    } else if (password === "") {
      setErrorMessage("Password is not provided.");
      return;
    } else if (invitationCode === "") {
      setErrorMessage("Invitation code is not provided.");
      return;
    }

    try {
      const response = await fetch(ENDPOINT_SIGNUP, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: username,
          password: md5(password),
          invitation_code: invitationCode,
        }),
      });

      if (response.status === 200) {
        navigate("/signin");
      } else {
        const data = await response.json();
        setErrorMessage(data.error);
      }
    } catch (err) {
      console.log(err);
      setErrorMessage("Something went wrong. Please try again later.");
    }
  }

  return (
    <div className="flex min-h-full flex-col justify-center px-6 py-12 lg:px-8">
      <div className="sm:mx-auto sm:w-full sm:max-w-sm">
        <img className="mx-auto h-20 w-auto" src={logoSrc} alt="Your Logo" />
        <h2 className="mt-10 text-center text-2xl font-bold leading-9 tracking-tight">
          Create a new account
        </h2>
      </div>

      <div className="mt-10 sm:mx-auto sm:w-full sm:max-w-sm">
        <form className="space-y-6" action="#" onSubmit={handleSignUp}>
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
            <div>
              <label
                htmlFor="password"
                className="block text-sm font-medium leading-6 text-balance"
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
              <p className="text-gray-600 text-xs mt-1">
                Recommend to contain 1 uppercase letter, 1 number, min. 8 characters.
              </p>
            </div>
          </div>

          <div>
            <div>
              <label
                htmlFor="invitation-code"
                className="block text-sm font-medium leading-6 text-black"
              >
                Invitation Code
              </label>
            </div>
            <div className="mt-2">
              <input
                id="invitation-code"
                name="invitation-code"
                type="invitation-code"
                autoComplete="off"
                onChange={handleInvitationCodeChange}
                className="block px-4 py-2 w-full rounded-md text-black border border-black focus:ring-0 focus:outline-none focus:border-2 sm:text-sm sm:leading-6 Input-class"
              />
            </div>
          </div>

          <div>
            <div>
              <p className="text-red-500 text-xs italic">{errorMessage}</p>
            </div>
            <button
              type="submit"
              className="flex mt-2 w-full justify-center rounded-md bg-red-500 px-4 py-2 text-sm font-semibold leading-6 text-white shadow-sm hover:bg-red-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-red-500"
            >
              Register
            </button>
          </div>
        </form>

        <p className="mt-10 text-center text-sm text-gray-400">
          Don't have an invitation code?{" "}
          <a
            href="https://yishenggong.com/about-me"
            className="font-semibold leading-6 text-red-500"
          >
            Contact with the author
          </a>
        </p>
      </div>
    </div>
  );
};

export default SignUp;
