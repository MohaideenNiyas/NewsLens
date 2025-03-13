document.addEventListener('DOMContentLoaded', function() {
    // Helper function to safely get DOM elements
    function safeGetElement(id) {
        const element = document.getElementById(id);
        if (!element) {
            console.warn(`Element with id '${id}' not found in the DOM. This may cause errors.`);
        }
        return element;
    }

    // DOM Elements - using safe getter
    const uploadArea = safeGetElement('upload-area');
    const fileInput = safeGetElement('file-input');
    const browseButton = safeGetElement('browse-button');
    const processButton = safeGetElement('process-button');
    const previewContainer = safeGetElement('preview-container');
    const previewImage = safeGetElement('preview-image');
    const loadingOverlay = safeGetElement('loading-overlay');
    const resultContainer = safeGetElement('result-container');
    const copyTextButton = safeGetElement('copy-text-button');
    const downloadJsonButton = safeGetElement('download-json-button');

    // Processing option checkboxes
    const enhanceContrastCheckbox = safeGetElement('enhance-contrast');
    const denoiseCheckbox = safeGetElement('denoise');
    const autoRotateCheckbox = safeGetElement('auto-rotate');
    const generateSummaryCheckbox = safeGetElement('generate-summary');
    const generateScriptCheckbox = safeGetElement('generate-script');

    // Tab elements
    const summaryTab = safeGetElement('summary-tab');
    const fullTextTab = safeGetElement('full-text-tab');
    const scriptTab = safeGetElement('script-tab');
    
    const summaryContent = safeGetElement('summary-content');
    const fullTextContent = safeGetElement('full-text-content');
    const scriptContent = safeGetElement('script-content');

    // Result elements - Summary tab
    const resultHeadline = safeGetElement('result-headline');
    const resultSource = safeGetElement('result-source');
    const resultLocation = safeGetElement('result-location');
    const resultDate = safeGetElement('result-date');
    const resultDateContainer = safeGetElement('result-date-container');
    const resultSeparator = safeGetElement('result-separator');
    const resultSummary = safeGetElement('result-summary');

    // Result elements - Full Text tab
    const resultFullHeadline = safeGetElement('result-full-headline');
    const resultFullSource = safeGetElement('result-full-source');
    const resultFullLocation = safeGetElement('result-full-location');
    const resultFullDate = safeGetElement('result-full-date');
    const resultFullDateContainer = safeGetElement('result-full-date-container');
    const resultFullSeparator = safeGetElement('result-full-separator');
    const resultBody = safeGetElement('result-body');

    // Result elements - Script tab
    const resultScript = safeGetElement('result-script');

    // Current data
    let currentFile = null;
    let extractedData = null;

    // Setup event listeners with null checks
    if (uploadArea) {
        uploadArea.addEventListener('click', function() {
            if (fileInput) fileInput.click();
        });

        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadArea.classList.add('bg-gray-100');
        });

        uploadArea.addEventListener('dragleave', function() {
            uploadArea.classList.remove('bg-gray-100');
        });

        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('bg-gray-100');
            
            if (e.dataTransfer.files.length) {
                handleFile(e.dataTransfer.files[0]);
            }
        });
    }

    if (browseButton) {
        browseButton.addEventListener('click', function(e) {
            e.stopPropagation();
            if (fileInput) fileInput.click();
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', function() {
            if (fileInput.files.length) {
                handleFile(fileInput.files[0]);
            }
        });
    }

    if (processButton) {
        processButton.addEventListener('click', processImage);
    }

    if (copyTextButton) {
        copyTextButton.addEventListener('click', copyExtractedText);
    }

    if (downloadJsonButton) {
        downloadJsonButton.addEventListener('click', downloadJsonData);
    }

    // Tab switching with null checks
    if (summaryTab) {
        summaryTab.addEventListener('click', function() {
            switchTab('summary-content');
        });
    }
    
    if (fullTextTab) {
        fullTextTab.addEventListener('click', function() {
            switchTab('full-text-content');
        });
    }
    
    if (scriptTab) {
        scriptTab.addEventListener('click', function() {
            switchTab('script-content');
        });
    }

    function switchTab(tabId) {
        // Guard clause to avoid errors if elements don't exist
        if (!summaryContent || !fullTextContent || !scriptContent ||
            !summaryTab || !fullTextTab || !scriptTab) {
            console.warn('Some tab elements missing - trying minimal tab switch');
            
            // Try a more minimal approach
            const targetContent = document.getElementById(tabId);
            if (targetContent) {
                // Hide all tab contents by class
                const allContents = document.querySelectorAll('.tab-content');
                allContents.forEach(content => content.classList.add('hidden'));
                
                // Show requested content
                targetContent.classList.remove('hidden');
                
                // Try to update tab styles if available
                const tabId = targetContent.id;
                const allTabs = document.querySelectorAll('[data-target]');
                allTabs.forEach(tab => {
                    tab.classList.remove('tab-active');
                    tab.classList.add('text-gray-500');
                    if (tab.getAttribute('data-target') === tabId) {
                        tab.classList.add('tab-active');
                        tab.classList.remove('text-gray-500');
                    }
                });
            }
            
            return;
        }

        // Full tab switching if all elements exist
        // Hide all tab contents
        summaryContent.classList.add('hidden');
        fullTextContent.classList.add('hidden');
        scriptContent.classList.add('hidden');
        
        // Show selected tab content
        const tabContent = document.getElementById(tabId);
        if (tabContent) {
            tabContent.classList.remove('hidden');
        }
        
        // Update tab styling
        summaryTab.classList.remove('tab-active');
        fullTextTab.classList.remove('tab-active');
        scriptTab.classList.remove('tab-active');
        summaryTab.classList.add('text-gray-500');
        fullTextTab.classList.add('text-gray-500');
        scriptTab.classList.add('text-gray-500');
        
        if (tabId === 'summary-content') {
            summaryTab.classList.add('tab-active');
            summaryTab.classList.remove('text-gray-500');
        } else if (tabId === 'full-text-content') {
            fullTextTab.classList.add('tab-active');
            fullTextTab.classList.remove('text-gray-500');
        } else if (tabId === 'script-content') {
            scriptTab.classList.add('tab-active');
            scriptTab.classList.remove('text-gray-500');
        }
    }

    // Handle the uploaded file
    function handleFile(file) {
        // Check if it's an image
        if (!file.type.match('image.*')) {
            alert('Please upload an image file');
            return;
        }

        currentFile = file;
        if (processButton) {
            processButton.disabled = false;
        }

        // Show preview
        if (previewImage && previewContainer) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.src = e.target.result;
                previewContainer.classList.remove('hidden');
                if (resultContainer) {
                    resultContainer.classList.add('hidden');
                }
            };
            reader.readAsDataURL(file);
        }
    }

    // Process the image
    function processImage() {
        if (!currentFile) return;

        // Show loading overlay
        if (loadingOverlay) {
            loadingOverlay.classList.remove('hidden');
        }
        
        // Get processing options from checkboxes
        const options = {
            enhanceContrast: enhanceContrastCheckbox ? enhanceContrastCheckbox.checked : true,
            denoise: denoiseCheckbox ? denoiseCheckbox.checked : true,
            autoRotate: autoRotateCheckbox ? autoRotateCheckbox.checked : true,
            extractStructure: true, // Always extract structure
            generateSummary: generateSummaryCheckbox ? generateSummaryCheckbox.checked : true,
            generateScript: generateScriptCheckbox ? generateScriptCheckbox.checked : true
        };
        
        // Create FormData
        const formData = new FormData();
        formData.append('image', currentFile);
        formData.append('options', JSON.stringify(options));

        // Send to backend
        fetch('/api/process', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Hide loading overlay
            if (loadingOverlay) {
                loadingOverlay.classList.add('hidden');
            }
            
            // Save extracted data
            extractedData = data;
            
            // Display the extracted data
            displayExtractedData(extractedData);
        })
        .catch(error => {
            console.error('Error processing image:', error);
            if (loadingOverlay) {
                loadingOverlay.classList.add('hidden');
            }
            alert('Error processing image: ' + error.message);
        });
    }

    // Display the extracted data
    function displayExtractedData(data) {
        if (!data) {
            console.error('No data to display');
            return;
        }

        // Always show Full Text tab regardless of content
        if (fullTextTab && fullTextTab.parentElement) {
            fullTextTab.parentElement.classList.remove('hidden');
        }

        // Set data for Summary tab
        if (resultHeadline) {
            resultHeadline.textContent = data.headline || "No headline detected";
        }
        if (resultSource) {
            resultSource.textContent = data.source || "";
        }
        if (resultLocation) {
            resultLocation.textContent = data.location || "";
        }
        if (resultSeparator) {
            resultSeparator.classList.toggle('hidden', !data.source && !data.location);
        }
        
        if (resultDate && resultDateContainer) {
            if (data.date) {
                resultDate.textContent = data.date;
                resultDateContainer.classList.remove('hidden');
            } else {
                resultDateContainer.classList.add('hidden');
            }
        }
        
        // Set summary if available
        if (resultSummary) {
            resultSummary.innerHTML = '';
            if (data.summary) {
                const p = document.createElement('p');
                p.textContent = data.summary;
                resultSummary.appendChild(p);
            } else if (data.body_text && data.body_text.length > 0) {
                // If no summary is available, use the first paragraph of the body text
                const p = document.createElement('p');
                p.textContent = data.body_text[0];
                resultSummary.appendChild(p);
            } else {
                const p = document.createElement('p');
                p.textContent = "No content available";
                p.classList.add('text-gray-500', 'italic');
                resultSummary.appendChild(p);
            }
        }
        
        // Populate the Full Text tab
        if (resultFullHeadline) {
            resultFullHeadline.textContent = data.headline || "No headline detected";
        }
        if (resultFullSource) {
            resultFullSource.textContent = data.source || "";
        }
        if (resultFullLocation) {
            resultFullLocation.textContent = data.location || "";
        }
        if (resultFullSeparator) {
            resultFullSeparator.classList.toggle('hidden', !data.source && !data.location);
        }
        
        if (resultFullDate && resultFullDateContainer) {
            if (data.date) {
                resultFullDate.textContent = data.date;
                resultFullDateContainer.classList.remove('hidden');
            } else {
                resultFullDateContainer.classList.add('hidden');
            }
        }
        
        // Set body text
        if (resultBody) {
            resultBody.innerHTML = '';
            if (data.body_text && data.body_text.length > 0) {
                data.body_text.forEach(paragraph => {
                    if (paragraph && paragraph.trim()) {
                        const p = document.createElement('p');
                        p.textContent = paragraph;
                        resultBody.appendChild(p);
                    }
                });
            } else {
                const p = document.createElement('p');
                p.textContent = "No article text detected";
                p.classList.add('text-gray-500', 'italic');
                resultBody.appendChild(p);
            }
        }
        
        // Set script if available
        if (resultScript) {
            resultScript.textContent = data.news_script || "Reporter script not available";
        }
        
        // Show/hide Script tab based on script availability and checkbox setting
        if (scriptTab && scriptTab.parentElement) {
            const shouldShowScript = data.news_script && (generateScriptCheckbox ? generateScriptCheckbox.checked : true);
            scriptTab.parentElement.classList.toggle('hidden', !shouldShowScript);
        }
        
        // Show results container and default to Summary tab
        if (resultContainer) {
            resultContainer.classList.remove('hidden');
        }
        
        try {
            // Default to the Summary tab if summary exists, otherwise show Full Text
            if (data.summary) {
                switchTab('summary-content');
            } else if (data.body_text && data.body_text.length > 0) {
                switchTab('full-text-content');
            } else if (data.news_script) {
                switchTab('script-content');
            } else {
                switchTab('summary-content');
            }
        } catch (e) {
            console.warn('Error switching tab:', e);
            
            // Fallback: manually show summary content if switchTab fails
            if (summaryContent) {
                document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
                summaryContent.classList.remove('hidden');
            }
        }
    }

    // Copy extracted text to clipboard
    function copyExtractedText() {
        if (!extractedData) return;
        
        // Determine which content is visible
        let textToCopy = "";
        
        const isSummaryVisible = !document.getElementById('summary-content')?.classList.contains('hidden');
        const isFullTextVisible = !document.getElementById('full-text-content')?.classList.contains('hidden');
        const isScriptVisible = !document.getElementById('script-content')?.classList.contains('hidden');
        
        if (isSummaryVisible) {
            // Copy summary
            textToCopy = (extractedData.headline || "") + "\n\n";
            if (extractedData.source) textToCopy += "Source: " + extractedData.source + "\n";
            if (extractedData.location) textToCopy += "Location: " + extractedData.location + "\n";
            if (extractedData.date) textToCopy += "Date: " + extractedData.date + "\n\n";
            if (extractedData.summary) textToCopy += "Summary:\n" + extractedData.summary;
            else if (extractedData.body_text && extractedData.body_text.length > 0) {
                textToCopy += "Summary:\n" + extractedData.body_text[0];
            }
        } 
        else if (isFullTextVisible) {
            // Copy full text
            textToCopy = (extractedData.headline || "") + "\n\n";
            if (extractedData.source) textToCopy += "Source: " + extractedData.source + "\n";
            if (extractedData.location) textToCopy += "Location: " + extractedData.location + "\n";
            if (extractedData.date) textToCopy += "Date: " + extractedData.date + "\n\n";
            
            if (extractedData.body_text && extractedData.body_text.length > 0) {
                extractedData.body_text.forEach(paragraph => {
                    if (paragraph && paragraph.trim()) {
                        textToCopy += paragraph + "\n\n";
                    }
                });
            }
        }
        else if (isScriptVisible) {
            // Copy script
            textToCopy = extractedData.news_script || "Reporter script not available";
        }
        else {
            // Fallback if no tab is explicitly visible
            textToCopy = "No content available to copy";
            
            // Check if we have a summary to copy as fallback
            if (extractedData.summary) {
                textToCopy = extractedData.summary;
            } else if (extractedData.news_script) {
                textToCopy = extractedData.news_script;
            } else if (extractedData.body_text && extractedData.body_text.length > 0) {
                textToCopy = extractedData.body_text.join("\n\n");
            }
        }
        
        navigator.clipboard.writeText(textToCopy)
            .then(() => {
                // Temporarily change button text to show success
                if (copyTextButton) {
                    const originalText = copyTextButton.innerHTML;
                    copyTextButton.innerHTML = '<i class="fas fa-check mr-1"></i> Copied!';
                    
                    setTimeout(() => {
                        copyTextButton.innerHTML = originalText;
                    }, 2000);
                }
            })
            .catch(err => {
                console.error('Failed to copy text: ', err);
                alert('Failed to copy text to clipboard');
            });
    }

    // Download JSON data
    function downloadJsonData() {
        if (!extractedData) return;
        
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(extractedData, null, 2));
        const downloadAnchor = document.createElement('a');
        downloadAnchor.setAttribute("href", dataStr);
        downloadAnchor.setAttribute("download", "newspaper_extraction.json");
        document.body.appendChild(downloadAnchor);
        downloadAnchor.click();
        downloadAnchor.remove();
    }
});