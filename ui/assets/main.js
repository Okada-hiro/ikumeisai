import {TranslationUI} from './ui-controller.js';
import {apiProcessor} from './api-processor.js';
import {waitUntilReady} from './boot.js';
import {LANGUAGES} from './languages.js';
await waitUntilReady();
new TranslationUI({processor:apiProcessor,maxRecordingMs:10000,languages:LANGUAGES});
