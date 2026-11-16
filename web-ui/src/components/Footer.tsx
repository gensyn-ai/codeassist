import { cssRecord } from "./cssRecord";

export function Footer() {
  return (
    <footer style={styles.self}>
      <div style={styles.links}>
        <a style={styles.link} href="https://docs.gensyn.ai/testnet/codeassist" target="_blank" rel="noopener noreferrer">Docs</a>
        <a style={styles.link} href="https://docs.gensyn.ai/testnet/codeassist/using-codeassist" target="_blank" rel="noopener noreferrer">Tutorial</a>
      </div>
    </footer>
  );
}

const styles = cssRecord({
  self: {
    display: "flex",
    padding: "28px",
  },
  links: {
    display: "flex",
    gap: "34px",
    fontSize: "14px",
  },
  link: {
    textDecoration: "none",
    color: "#ffffff",
  },
});
