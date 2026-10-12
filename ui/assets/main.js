import {TranslationUI} from './ui-controller.js';
import {apiProcessor} from './api-processor.js';
import {waitUntilReady} from './boot.js';
await waitUntilReady();
new TranslationUI({processor:apiProcessor,maxRecordingMs:10000});
