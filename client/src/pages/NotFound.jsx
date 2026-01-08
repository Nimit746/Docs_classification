/* eslint-disable no-unused-vars */
import React from 'react';
import { Link } from 'react-router-dom';
import { Home, AlertCircle } from 'lucide-react';
import { motion } from 'framer-motion';

const NotFound = () => {
    return (
        <div className="flex flex-col items-center justify-center min-h-[60vh] text-center px-4">
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="max-w-md"
            >
                <div className="bg-red-50 text-red-500 p-4 rounded-full w-20 h-20 flex items-center justify-center mx-auto mb-6">
                    <AlertCircle size={40} />
                </div>
                <h1 className="text-4xl font-bold text-[var(--color-text-main)] mb-4">Page Not Found</h1>
                <p className="text-[var(--color-text-muted)] mb-8 text-lg">
                    The page you are looking for doesn't exist or has been moved.
                </p>
                <Link
                    to="/"
                    className="inline-flex items-center gap-2 px-6 py-3 bg-[var(--color-primary)] text-white rounded-xl font-medium hover:bg-blue-600 transition-colors shadow-lg shadow-blue-500/20"
                >
                    <Home size={20} />
                    Back to Home
                </Link>
            </motion.div>
        </div>
    );
};

export default NotFound;
