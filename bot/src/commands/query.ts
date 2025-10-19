import {
  ChatInputCommandInteraction,
  SlashCommandBuilder,
  AttachmentBuilder,
  EmbedBuilder,
} from "discord.js";
import { Config, GeneratePayload, GenerateResponse } from "../types";
import { fetchJson } from "../utils/utils";
import https from "https";
import http from "http";

export const data = new SlashCommandBuilder()
  .setName("query")
  .setDescription("Send a prompt to MeowerAI and receive generated text.")
  .addStringOption((option) =>
    option
      .setName("prompt")
      .setDescription("The prompt to send to the model")
      .setRequired(true),
  );

export async function execute(
  interaction: ChatInputCommandInteraction,
  config: Config,
  agent: https.Agent | http.Agent | null,
): Promise<void> {
  const prompt = interaction.options.getString("prompt", true);

  const thinkingEmbed = new EmbedBuilder()
    .setColor(0x5865f2)
    .setTitle("🧠 MeowerAI is thinking...")
    .setDescription(
      `\`\`\`\n${prompt.substring(0, 200)}${prompt.length > 200 ? "..." : ""}\n\`\`\``,
    )
    .addFields(
      {
        name: "⚙️ Temperature",
        value: `\`${config.llm.temperature}\``,
        inline: true,
      },
      { name: "🎲 Top-P", value: `\`${config.llm.top_p}\``, inline: true },
      {
        name: "📊 Max Tokens",
        value: `\`${config.llm.maxTokens}\``,
        inline: true,
      },
    )
    .setFooter({ text: "Generating response..." })
    .setTimestamp();

  await interaction.reply({ embeds: [thinkingEmbed] });

  try {
    const llm = config.llm;
    const startTime = Date.now();

    const payload: GeneratePayload = {
      prompt,
      max_new_tokens: llm.maxTokens,
      temperature: llm.temperature,
      top_p: llm.top_p,
      do_sample: llm.do_sample,
    };

    const genUrl = `${llm.endpoint}/generate`;
    console.log(`[query] Generating with prompt: "${prompt.substring(0, 50)}..."`);

    const res = await fetchJson<GenerateResponse>(genUrl, llm.apiKey, agent, {
      method: "POST",
      body: JSON.stringify(payload),
    });

    const generationTime = ((Date.now() - startTime) / 1000).toFixed(2);

    if (!res.ok) {
      const bodyText =
        typeof res.body === "object"
          ? JSON.stringify(res.body)
          : String(res.body);
      throw new Error(`API returned ${res.status}: ${bodyText}`);
    }

    const responseBody = res.body;
    let outText: string;

    if (typeof responseBody === "string") {
      outText = responseBody;
    } else if (
      responseBody &&
      typeof responseBody === "object" &&
      responseBody.text !== undefined
    ) {
      outText = responseBody.text;
    } else {
      outText = JSON.stringify(responseBody, null, 2);
    }

    const tokensGenerated = responseBody.tokens_generated ?? 0;
    const tokensPerSecond =
      tokensGenerated > 0
        ? (tokensGenerated / parseFloat(generationTime)).toFixed(2)
        : "N/A";

    if (outText.length > 1900) {
      const filename = `meowerai_${Date.now()}.txt`;
      const buffer = Buffer.from(outText);
      const attachment = new AttachmentBuilder(buffer, { name: filename });

      const resultEmbed = new EmbedBuilder()
        .setColor(0x57f287)
        .setTitle("✨ Response Generated Successfully")
        .setDescription(
          "Response was too long and has been attached as a file.",
        )
        .addFields(
          {
            name: "📝 Tokens Generated",
            value: `\`${tokensGenerated}\``,
            inline: true,
          },
          {
            name: "⚡ Generation Time",
            value: `\`${generationTime}s\``,
            inline: true,
          },
          {
            name: "🚀 Speed",
            value: `\`${tokensPerSecond} tok/s\``,
            inline: true,
          },
          {
            name: "💬 Your Prompt",
            value: `\`\`\`\n${prompt.substring(0, 100)}${prompt.length > 100 ? "..." : ""}\n\`\`\``,
            inline: false,
          },
        )
        .setFooter({ text: `${llm.name} • ${llm.version}` })
        .setTimestamp();

      await interaction.editReply({
        embeds: [resultEmbed],
        files: [attachment],
      });
    } else {
      const resultEmbed = new EmbedBuilder()
        .setColor(0x57f287)
        .setTitle("✨ Response Generated")
        .setDescription(`\`\`\`\n${outText}\n\`\`\``)
        .addFields(
          { name: "📝 Tokens", value: `\`${tokensGenerated}\``, inline: true },
          { name: "⚡ Time", value: `\`${generationTime}s\``, inline: true },
          {
            name: "🚀 Speed",
            value: `\`${tokensPerSecond} tok/s\``,
            inline: true,
          },
        )
        .setFooter({ text: `${llm.name} • ${llm.version}` })
        .setTimestamp();

      await interaction.editReply({ embeds: [resultEmbed] });
    }
  } catch (err: any) {
    console.error("query error:", err);

    let errorMsg = String(err.message || err);
    let troubleshootingTips = "";

    if (errorMsg.includes("EPROTO") || errorMsg.includes("SSL")) {
      troubleshootingTips =
        "🔧 Make sure your endpoint uses `http://` not `https://`";
    } else if (errorMsg.includes("ECONNREFUSED")) {
      troubleshootingTips =
        "🔧 Is the API server running? Check the health endpoint";
    } else if (errorMsg.includes("timeout")) {
      troubleshootingTips =
        "🔧 The request timed out. Try a shorter prompt or check server load";
    }

    const errorEmbed = new EmbedBuilder()
      .setColor(0xed4245)
      .setTitle("❌ Generation Failed")
      .setDescription(`\`\`\`\n${errorMsg.substring(0, 500)}\n\`\`\``)
      .addFields({
        name: "💡 Troubleshooting",
        value: troubleshootingTips || "Check the console logs for more details",
      })
      .setFooter({ text: "MeowerAI Error Handler" })
      .setTimestamp();

    await interaction.editReply({ embeds: [errorEmbed] });
  }
}
