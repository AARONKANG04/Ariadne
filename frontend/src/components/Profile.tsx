import { useAuth0 } from "@auth0/auth0-react";

const Profile = () => {
  const { user, isAuthenticated, isLoading } = useAuth0();

  if (isLoading) {
    return <div className="text-xl text-slate-300 font-medium">Loading profile...</div>;
  }

  const defaultAvatar = `data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='110' height='110' viewBox='0 0 110 110'%3E%3Ccircle cx='55' cy='55' r='55' fill='%2363b3ed'/%3E%3Cpath d='M55 50c8.28 0 15-6.72 15-15s-6.72-15-15-15-15 6.72-15 15 6.72 15 15 15zm0 7.5c-10 0-30 5.02-30 15v3.75c0 2.07 1.68 3.75 3.75 3.75h52.5c2.07 0 3.75-1.68 3.75-3.75V72.5c0-9.98-20-15-30-15z' fill='%23fff'/%3E%3C/svg%3E`;

  return (
    isAuthenticated && user ? (
      <div className="flex flex-col items-center gap-4">
        <img 
          src={user.picture || defaultAvatar} 
          alt={user.name || 'User'} 
          className="w-[110px] h-[110px] rounded-full object-cover border-[3px] border-blue-400"
          onError={(e) => {
            const target = e.target as HTMLImageElement;
            target.src = defaultAvatar;
          }}
        />
        <div className="text-center">
          <div className="text-3xl font-semibold text-slate-50 mb-2">
            {user.name}
          </div>
          <div className="text-lg text-slate-400">
            {user.email}
          </div>
        </div>
      </div>
    ) : null
  );
};

export default Profile;
