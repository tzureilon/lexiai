import type { AppProps } from "next/app";

import { AuthProvider } from "../lib/auth";
import "../styles/globals.css";

export default function LexiAIApp({ Component, pageProps }: AppProps) {
  return (
    <AuthProvider>
      <Component {...pageProps} />
    </AuthProvider>
  );
}
