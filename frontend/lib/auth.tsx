"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/router";

import {
  Session,
  clearSession as clearStoredSession,
  getSession as getStoredSession,
  storeSession,
} from "./session";

type AuthContextValue = {
  session: Session | null;
  loading: boolean;
  login: (token: string) => void;
  logout: () => void;
};

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setSession(getStoredSession());
    setLoading(false);
  }, []);

  const login = useCallback((token: string) => {
    const nextSession = storeSession(token);
    setSession(nextSession);
  }, []);

  const logout = useCallback(() => {
    clearStoredSession();
    setSession(null);
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({ session, loading, login, logout }),
    [loading, login, logout, session],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}

export function useRequireAuth() {
  const { session, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (loading) {
      return;
    }
    if (!session) {
      const next = typeof window !== "undefined" ? window.location.pathname + window.location.search : "/";
      void router.replace(`/login?next=${encodeURIComponent(next)}`);
    }
  }, [loading, router, session]);

  return { session, loading };
}
