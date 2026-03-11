import {env} from "./constants";
import {createOpenAI} from "@ai-sdk/openai";
import {createAnthropic} from "@ai-sdk/anthropic";
import {experimental_transcribe, generateText, Output} from "ai";
import {z} from "zod";
import {pipeline} from '@huggingface/transformers';
import {WaveFile} from 'wavefile';

function getOpenAIClient() {
    if (!env.OPENAI_API_KEY) return null;
    return createOpenAI({ baseURL: env.OPENAI_URL, apiKey: env.OPENAI_API_KEY });
}

function getTextModel() {
    if (env.ANTHROPIC_API_KEY) {
        return createAnthropic({ apiKey: env.ANTHROPIC_API_KEY })(env.TEXT_MODEL);
    }
    const openai = getOpenAIClient();
    if (openai) return openai.chat(env.TEXT_MODEL);
    throw new Error("No AI provider configured: set ANTHROPIC_API_KEY or OPENAI_API_KEY");
}

export async function getTranscription(blob: Blob): Promise<string> {
    if (env.LOCAL_TRANSCRIPTION_MODEL) {
        console.info("Using local Whisper model for transcription:", env.LOCAL_TRANSCRIPTION_MODEL);
        const transcriber = await pipeline('automatic-speech-recognition', env.LOCAL_TRANSCRIPTION_MODEL);
        const arrayBuffer = Buffer.from(await blob.arrayBuffer());
        try {
            const wav = new WaveFile(new Uint8Array(arrayBuffer));
            wav.toBitDepth('32f');
            wav.toSampleRate(16000);
            let audioData: any = wav.getSamples();
            const result = await transcriber(audioData, {
                chunk_length_s: 30,
                stride_length_s: 5,
            });

            if (result && typeof result === 'object' && 'text' in result) {
                return (result as any).text;
            }

            return String(result);
        } catch (err) {
            console.error('Error transcribing with local Whisper model:', err);
            throw err;
        }
    }

    const openaiClient = getOpenAIClient();
    if (!openaiClient) {
        throw new Error("No transcription provider available: set LOCAL_TRANSCRIPTION_MODEL or OPENAI_API_KEY");
    }

    try {
        const audioBuffer = Buffer.from(await blob.arrayBuffer());

        const result = await experimental_transcribe({
            model: openaiClient.transcription(env.TRANSCRIPTION_MODEL),
            audio: audioBuffer,
        });

        return result.text;
    } catch (error) {
        console.error("Error in getTranscription (AI SDK):", error);
        throw new Error("Failed to transcribe audio via API");
    }
}

export async function generateRecipeFromAI(
    transcription: string,
    description: string,
    postURL: string,
    thumbnail: string,
    extraPrompt: string,
    tags: string[],
    images: string[],
) {
    const schema = Output.object({
        schema: z.object({
            "@context": z
                .literal("https://schema.org")
                .default("https://schema.org"),
            "@type": z.literal("Recipe").default("Recipe"),
            name: z.string(),
            image: z.string().optional(),
            url: z.string().optional(),
            description: z.string(),
            recipeIngredient: z.array(z.string()),
            recipeInstructions: z.array(
                z.object({
                    "@type": z.literal("HowToStep").default("HowToStep"),
                    text: z.string(),
                })
            ),
            keywords: z.array(z.string()).optional(),
            nutrition: z.object({
                "@type": z.literal("NutritionInformation").default("NutritionInformation"),
                calories: z.string().optional(),
                carbohydrateContent: z.string().optional(),
                proteinContent: z.string().optional(),
                fatContent: z.string().optional(),
                cholesterolContent: z.string().optional(),
                fiberContent: z.string().optional(),
                sugarContent: z.string().optional(),
                sodiumContent: z.string().optional(),
            }).optional()
        }),
    });

    console.info("generateRecipeFromAI called, using model:", env.ANTHROPIC_API_KEY ? `anthropic/${env.TEXT_MODEL}` : `openai/${env.TEXT_MODEL}`);
    try {
        const userPrompt = `<Metadata>
            Post URL: ${postURL}
            Description: ${description}
            Thumbnail: ${thumbnail}
        </Metadata>

        <Transcription>
        ${transcription}
        </Transcription>

        ${
            tags && tags.length > 0 && Array.isArray(tags)
                ? `<keywords>${tags.join(", ")}</keywords>`
                : ""
        }

        ${
            tags && tags.length > 0 && !Array.isArray(tags)
                ? `<keywords>${tags}</keywords>`
                : ""
        }

        Use the thumbnail for the image field and the post URL for the url field.
        Extract ingredients and instructions clearly.
        Output must be valid JSON-LD Schema.org Recipe format.
        ${
            extraPrompt.length > 1
                ? ` Also the user requests that:
        ${extraPrompt}`
                : ""
        }
        `;

        console.info("Calling generateText with", images.filter(img => img).length, "images");
        const {output} = await generateText({
            model: getTextModel(),
            output: schema,
            messages: [
                {
                    role: "system",
                    content: "You are an expert chef assistant. Review the following recipe transcript and refine it for clarity, conciseness, and accuracy.\n" +
                        "Ensure ingredients and instructions are well-formatted and easy to follow.\n" +
                        "Correct any obvious errors or omissions.\n" +
                        "Output must be valid JSON-LD Schema.org Recipe format.\n" +
                        "The keywords field should not be modified leave it as it comes, it they are not present dont include them. Only add relevant tags dont add tags that are not relevant to the recipe."
                },
                {
                    role: "user",
                    content: [
                        {
                            type: "text" as const,
                            text: userPrompt,
                        },
                        ...images.filter(img => img).map(img => ({
                            type: "image" as const,
                            image: img,
                        }))
                    ],
                }
            ],
        });
        return output;
    } catch
        (error) {
        console.error("Error generating recipe with AI:", error);
        throw new Error("Failed to generate recipe structure");
    }
}