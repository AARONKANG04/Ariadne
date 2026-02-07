import { useAuth0 } from "@auth0/auth0-react";

const LogoutButton = () => {
  const { logout } = useAuth0();
  return (
    <button
      onClick={() => logout({ logoutParams: { returnTo: window.location.origin } })}
      className="px-6 py-2.5 text-sm font-medium rounded-lg border border-white text-white bg-transparent cursor-pointer transition-all duration-200 ease-out outline-none hover:bg-white/10 focus:ring-2 focus:ring-white/50"
    >
      Log Out
    </button>
  );
};

export default LogoutButton;
