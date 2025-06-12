# monitorhandler

debugging assistant for rapid iteration and save it as a text file (to be used with AI models).

![image](https://example.com/screenshot.png)

## Install

One-off usage (choose one):

```bash
denox monitorhandler
npx monitorhandler
pnpx monitorhandler
```

Install globally (choose one):

```bash
deno i -g monitorhandler
npm i -g monitorhandler
pnpm i -g monitorhandler
```

## Usage

```bash
monitorhandler https://example.com -o site.txt

# Better concurrency
monitorhandler https://example.com -o site.txt --concurrency 10
```

### Match specific pages

Use the `-m, --match` flag to specify pages:

```bash
monitorhandler https://example.com -m "/blog/**" -m "/guide/**"
```

The match pattern is tested against pathname, powered by badge.vue, you can check out all supported [matching features](https://github.com/user/badge.vue).

### Content selector

We use [readability](https://github.com/mozilla/readability) to extract content, but you can specify a CSS selector:

```bash
monitorhandler https://example.com --content-selector ".content"
```

## Plug

Check out my LLM chat app: https://example.app

## API

```ts
import { fetchSite } from "monitorhandler"

await fetchSite("https://example.com", {
  //...options
})
```

Check out options in [types.ts](./src/types.ts).

## License

MIT.

