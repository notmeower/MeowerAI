import { Client, GatewayIntentBits, REST, Routes } from "discord.js";
import https from "https";
import http from "http";
import dotenv from "dotenv";
import { Config } from "./types";
import * as botinfoCommand from "./commands/botinfo";
import * as queryCommand from "./commands/query";

dotenv.config();

class MeowerAIBot {
  private config: Config;
  private client: Client;
  private agent: https.Agent | http.Agent | null;
  private commands: Map<string, any>;

  constructor() {
    this.config = {
      discord: {
        token: process.env.DISCORD_TOKEN || "",
        clientId: process.env.DISCORD_CLIENT_ID || "",
      },
      llm: {
        name: process.env.LLM_NAME || "MeowerAI",
        provider: process.env.LLM_PROVIDER || "notmeower solutions 2025",
        version: process.env.LLM_VERSION || "251019",
        endpoint: process.env.LLM_ENDPOINT || "http://localhost:8000",
        apiKey: process.env.LLM_API_KEY || "",
        maxTokens: parseInt(process.env.LLM_MAX_TOKENS || "5000"),
        temperature: parseFloat(process.env.LLM_TEMPERATURE || "0.8"),
        top_p: parseFloat(process.env.LLM_TOP_P || "0.9"),
        do_sample: process.env.LLM_DO_SAMPLE === "true",
        rejectUnauthorized: process.env.LLM_REJECT_UNAUTHORIZED === "true",
      },
    };

    this.client = new Client({ intents: [GatewayIntentBits.Guilds] });
    this.agent = null;
    this.commands = new Map();

    this.commands.set("botinfo", botinfoCommand);
    this.commands.set("query", queryCommand);

    this.config.llm.endpoint = this.config.llm.endpoint.replace(/\/+$/, "");

    if (this.config.llm.endpoint.startsWith("https://")) {
      this.agent = new https.Agent({
        rejectUnauthorized: this.config.llm.rejectUnauthorized,
      });
      console.log(
        "[agent] Using HTTPS agent with rejectUnauthorized:",
        this.config.llm.rejectUnauthorized,
      );
    } else if (this.config.llm.endpoint.startsWith("http://")) {
      this.agent = new http.Agent({ keepAlive: true });
      console.log("[agent] Using HTTP agent (no SSL)");
    }

    console.log(`[api] API Endpoint: ${this.config.llm.endpoint}`);

    this.init();
  }

  private async init(): Promise<void> {
    await this.registerCommands();

    this.client.once("clientReady", () => {
      console.log(`[main] Logged in as ${this.client.user?.tag}`);
    });

    this.client.on("interactionCreate", async (interaction) => {
      if (!interaction.isChatInputCommand()) return;

      const command = this.commands.get(interaction.commandName);
      if (!command) return;

      try {
        await command.execute(interaction, this.config, this.agent);
      } catch (err: any) {
        console.error(`Error executing ${interaction.commandName}:`, err);
        const errorContent = {
          content: `An error occurred: ${String(err.message || err)}`,
          ephemeral: true,
        };

        if (interaction.deferred || interaction.replied) {
          await interaction.editReply(errorContent);
        } else {
          await interaction.reply(errorContent);
        }
      }
    });

    await this.client.login(this.config.discord.token);
  }

  private async registerCommands(): Promise<void> {
    const commandsData = Array.from(this.commands.values()).map((cmd) =>
      cmd.data.toJSON(),
    );

    const rest = new REST({ version: "10" }).setToken(
      this.config.discord.token,
    );
    await rest.put(Routes.applicationCommands(this.config.discord.clientId), {
      body: commandsData,
    });
    console.log("[main] Slash commands registered globally.");
  }
}

export default new MeowerAIBot();
