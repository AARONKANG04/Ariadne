import { useAuth0 } from "@auth0/auth0-react";

const LoginButton = () => {
  const { loginWithRedirect } = useAuth0();
  return (
    <button 
      onClick={() => loginWithRedirect()} 
      className="px-8 py-4 text-base font-semibold rounded-lg border-none cursor-pointer transition-all duration-300 ease-out outline-none uppercase tracking-wider shadow-lg bg-gradient-to-r from-blue-400 to-blue-500 text-slate-900 hover:transform hover:-translate-y-1 hover:scale-105 hover:shadow-xl focus:ring-4 focus:ring-blue-400/50"
    >
      Log In
    </button>
  );
};

export default LoginButton;
