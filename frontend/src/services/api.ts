import axios from 'axios';

const api = axios.create({
    baseURL: '/', // Proxy will handle this
    headers: {
        'Content-Type': 'application/json',
    },
});

export interface TranslationResponse {
    translation?: string;
    error?: string;
}

export const translateText = async (text: string): Promise<string> => {
    try {
        const response = await api.post<TranslationResponse>('/translate', { text });
        if (response.data.error) {
            throw new Error(response.data.error);
        }
        return response.data.translation || '';
    } catch (error) {
        if (axios.isAxiosError(error)) {
            const msg = error.response?.data?.detail || error.response?.data?.error;
            if (msg) throw new Error(msg);
        }
        throw new Error('Failed to connect to translation service.');
    }
};
