import { ChatInputCommandInteraction, SlashCommandBuilder } from "discord.js";
import { Config, HealthResponse, InfoResponse } from "../types";
import { fetchJson, formatNumber } from "../utils/utils";
import https from "https";
import http from "http";

export const data = new SlashCommandBuilder()
  .setName("botinfo")
  .setDescription(
    "Show information and status about the MeowerAI model and API.",
  );

export async function execute(
  interaction: ChatInputCommandInteraction,
  config: Config,
  agent: https.Agent | http.Agent | null,
): Promise<void> {
  await interaction.deferReply().catch(() => {});

  try {
    const llm = config.llm;
    const healthUrl = `${llm.endpoint}/health`;
    const infoUrl = `${llm.endpoint}/info`;

    const pingStart = Date.now();
    const [healthRes, infoRes] = await Promise.all([
      fetchJson<HealthResponse>(healthUrl, llm.apiKey, agent, {
        method: "GET",
      }),
      fetchJson<InfoResponse>(infoUrl, llm.apiKey, agent, { method: "GET" }),
    ]);
    const pingMs = Date.now() - pingStart;

    const status =
      healthRes.ok && healthRes.body && healthRes.body.model_loaded
        ? "🟢 ONLINE"
        : "🔴 OFFLINE";

    let infoText = "```\n";
    infoText += "╔═══════════════════════════════════════════════╗\n";
    infoText += "║           MEOWERAI - SYSTEM STATUS            ║\n";
    infoText += "╚═══════════════════════════════════════════════╝\n";
    infoText += "```\n\n";

    infoText += "```ansi\n";
    infoText += "┌─ MODEL INFORMATION\n";
    infoText += `│ Name:        ${llm.name ?? "unknown"}\n`;
    infoText += `│ Provider:    ${llm.provider}\n`;
    infoText += `│ Version:     ${llm.version ?? "unknown"}\n`;
    infoText += `│ Status:      ${status}\n`;
    infoText += "└─\n";
    infoText += "```\n\n";

    infoText += "```ansi\n";
    infoText += "┌─ API CONFIGURATION\n";
    infoText += `│ Endpoint:    ${llm.endpoint}\n`;
    infoText += `│ Ping:        ${pingMs}ms\n`;
    infoText += `│ Max Tokens:  ${llm.maxTokens}\n`;
    infoText += `│ Temperature: ${llm.temperature}\n`;
    infoText += `│ Top-P:       ${llm.top_p}\n`;
    infoText += `│ Sampling:    ${llm.do_sample ? "enabled" : "disabled"}\n`;
    infoText += "└─\n";
    infoText += "```\n";

    if (infoRes.ok && infoRes.body && typeof infoRes.body === "object") {
      const info = infoRes.body;
      infoText += "\n```ansi\n";
      infoText += "┌─ TECHNICAL SPECIFICATIONS\n";
      if (info.parameters !== undefined) {
        infoText += `│ Parameters:  ${formatNumber(info.parameters)}\n`;
      }
      if (info.max_seq_length !== undefined) {
        infoText += `│ Max Length:  ${info.max_seq_length.toLocaleString()}\n`;
      }
      if (info.vocab_size !== undefined) {
        infoText += `│ Vocab Size:  ${formatNumber(info.vocab_size)}\n`;
      }
      if (info.device !== undefined) {
        infoText += `│ Device:      ${info.device.toUpperCase()}\n`;
      }
      infoText += "└─\n";
      infoText += "```";
    } else if (!infoRes.ok) {
      infoText += `\n\`\`\`diff\n- Failed to fetch technical specs (HTTP ${infoRes.status})\n\`\`\``;
    }

    await interaction.editReply({ content: infoText });
  } catch (err: any) {
    console.error("botinfo error:", err);
    await interaction.editReply({
      content: `❌ Error fetching LLM info: ${String(err.message || err)}`,
    });
  }
}
