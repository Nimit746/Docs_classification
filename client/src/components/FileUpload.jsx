/* eslint-disable no-unused-vars */
import React, { useRef, useState } from 'react';
import { Upload, File, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const FileUpload = ({ onFileSelect, selectedFile, onClear }) => {
    const [isDragging, setIsDragging] = useState(false);
    const fileInputRef = useRef(null);

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        setIsDragging(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            validateAndSelect(files[0]);
        }
    };

    const handleFileInput = (e) => {
        if (e.target.files.length > 0) {
            validateAndSelect(e.target.files[0]);
        }
    };

    const validateAndSelect = (file) => {
        // Add validation logic here if needed (e.g., file type, size)
        onFileSelect(file);
    };

    return (
        <div className="w-full max-w-2xl mx-auto">
            <AnimatePresence mode="wait">
                {!selectedFile ? (
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        key="upload-zone"
                        className={`
                            relative border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all duration-300
                            ${isDragging
                                ? 'border-[var(--color-primary)] bg-[var(--color-primary-light)]/20'
                                : 'border-slate-300 hover:border-[var(--color-primary)] hover:bg-slate-50'
                            }
                        `}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        onClick={() => fileInputRef.current?.click()}
                    >
                        <input
                            type="file"
                            ref={fileInputRef}
                            onChange={handleFileInput}
                            className="hidden"
                            accept=".pdf,.docx,.doc,.txt"
                        />

                        <div className="flex flex-col items-center gap-4">
                            <div className={`
                                p-4 rounded-full transition-colors duration-300
                                ${isDragging ? 'bg-[var(--color-primary)] text-white' : 'bg-slate-100 text-[var(--color-text-muted)] group-hover:bg-[var(--color-primary)] group-hover:text-white'}
                            `}>
                                <Upload size={32} />
                            </div>
                            <div>
                                <h3 className="text-lg font-semibold text-[var(--color-text-main)] mb-1">
                                    Click to upload or drag and drop
                                </h3>
                                <p className="text-sm text-[var(--color-text-muted)]">
                                    Supported formats: PDF, DOCX, TXT
                                </p>
                            </div>
                        </div>
                    </motion.div>
                ) : (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        key="file-preview"
                        className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm flex items-center justify-between"
                    >
                        <div className="flex items-center gap-4">
                            <div className="p-3 bg-[var(--color-primary-light)] text-[var(--color-primary)] rounded-lg">
                                <File size={24} />
                            </div>
                            <div className="text-left">
                                <h4 className="font-medium text-[var(--color-text-main)] truncate max-w-[200px] sm:max-w-md">
                                    {selectedFile.name}
                                </h4>
                                <p className="text-xs text-[var(--color-text-muted)]">
                                    {(selectedFile.size / 1024).toFixed(2)} KB
                                </p>
                            </div>
                        </div>
                        <button
                            onClick={(e) => {
                                e.stopPropagation();
                                onClear();
                            }}
                            className="p-2 text-slate-400 hover:text-[var(--color-error)] hover:bg-red-50 rounded-full transition-colors"
                        >
                            <X size={20} />
                        </button>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default FileUpload;
