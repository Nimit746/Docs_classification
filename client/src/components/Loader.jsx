import React from 'react';

/**
 * Versatile Loader Component Library
 * 
 * Usage Examples:
 * <Loader />                                    // Default spinner
 * <Loader type="dots" />                        // Bouncing dots
 * <Loader type="pulse" size="lg" />             // Large pulse loader
 * <Loader type="bars" color="red" />            // Red bars loader
 * <Loader type="ring" color="blue" size="md" /> // Medium blue ring
 * <Loader type="dual-ring" />                   // Dual ring spinner
 * <Loader type="ripple" />                      // Ripple effect
 * <Loader type="grid" />                        // Grid loader
 * 
 * Props:
 * - type: 'spinner' | 'dots' | 'pulse' | 'bars' | 'ring' | 'dual-ring' | 'ripple' | 'grid'
 * - size: 'sm' | 'md' | 'lg' | 'xl' (default: 'md')
 * - color: string (Tailwind color class like 'blue', 'red', 'green', etc.)
 * - className: additional CSS classes
 * - fullScreen: boolean - centers loader on full screen with backdrop
 * - text: string - optional loading text below loader
 */

const Loader = ({
    type = 'spinner',
    size = 'md',
    color = 'blue',
    className = '',
    fullScreen = false,
    text = ''
}) => {
    // Color presets mapping
    const colorMap = {
        blue: '#3b82f6',
        red: '#ef4444',
        green: '#22c55e',
        purple: '#a855f7',
        pink: '#ec4899',
        yellow: '#eab308',
        indigo: '#6366f1',
        gray: '#6b7280',
        orange: '#f97316',
        teal: '#14b8a6',
        cyan: '#06b6d4',
        lime: '#84cc16',
        emerald: '#10b981',
        violet: '#8b5cf6',
        fuchsia: '#d946ef',
        rose: '#f43f5e',
        slate: '#64748b',
        zinc: '#71717a',
        neutral: '#737373',
        stone: '#78716c',
        amber: '#f59e0b',
        sky: '#0ea5e9'
    };

    const colorValue = colorMap[color] || colorMap.blue;

    // Size mappings
    const sizes = {
        sm: { width: 'w-4 h-4', text: 'text-xs', gap: 'gap-1', px: 16, dotPx: 8 },
        md: { width: 'w-8 h-8', text: 'text-sm', gap: 'gap-2', px: 32, dotPx: 12 },
        lg: { width: 'w-12 h-12', text: 'text-base', gap: 'gap-3', px: 48, dotPx: 16 },
        xl: { width: 'w-16 h-16', text: 'text-lg', gap: 'gap-4', px: 64, dotPx: 20 }
    };

    const currentSize = sizes[size] || sizes.md;

    // Loader components
    const loaders = {
        spinner: (
            <div
                className={`${currentSize.width} border-4 border-gray-200 border-t-transparent rounded-full animate-spin`}
                style={{ borderTopColor: colorValue }}
            />
        ),

        dots: (
            <div className={`flex ${currentSize.gap}`}>
                {[0, 1, 2].map((i) => (
                    <div
                        key={i}
                        className={`${currentSize.width.split(' ')[0]} ${currentSize.width.split(' ')[1]} rounded-full animate-bounce`}
                        style={{ backgroundColor: colorValue, animationDelay: `${i * 0.15}s` }}
                    />
                ))}
            </div>
        ),

        pulse: (
            <div className="relative">
                <div
                    className={`${currentSize.width} rounded-full animate-ping absolute opacity-75`}
                    style={{ backgroundColor: colorValue }}
                />
                <div
                    className={`${currentSize.width} rounded-full relative`}
                    style={{ backgroundColor: colorValue }}
                />
            </div>
        ),

        bars: (
            <div className={`flex items-end ${currentSize.gap}`}>
                {[0, 1, 2, 3, 4].map((i) => (
                    <div
                        key={i}
                        className={`w-1 rounded-full animate-pulse`}
                        style={{
                            backgroundColor: colorValue,
                            height: size === 'sm' ? '16px' : size === 'lg' ? '48px' : size === 'xl' ? '64px' : '32px',
                            animationDelay: `${i * 0.1}s`,
                            animationDuration: '1s'
                        }}
                    />
                ))}
            </div>
        ),

        ring: (
            <div className="relative">
                <div
                    className={`${currentSize.width} border-4 border-gray-200 rounded-full`}
                />
                <div
                    className={`${currentSize.width} border-4 border-t-transparent rounded-full animate-spin absolute inset-0`}
                    style={{ borderColor: colorValue, borderTopColor: 'transparent' }}
                />
            </div>
        ),

        'dual-ring': (
            <div className="relative">
                <div
                    className={`${currentSize.width} border-4 border-t-transparent rounded-full animate-spin`}
                    style={{ borderColor: colorValue, borderTopColor: 'transparent' }}
                />
                <div
                    className={`${currentSize.width} border-4 border-gray-300 border-b-transparent rounded-full animate-spin absolute inset-0`}
                    style={{ animationDirection: 'reverse', animationDuration: '0.8s' }}
                />
            </div>
        ),

        ripple: (
            <div className={`relative ${currentSize.width}`}>
                {[0, 1].map((i) => (
                    <div
                        key={i}
                        className={`absolute inset-0 border-4 rounded-full animate-ping`}
                        style={{
                            borderColor: colorValue,
                            animationDelay: `${i * 0.5}s`,
                            animationDuration: '1.5s'
                        }}
                    />
                ))}
            </div>
        ),

        grid: (
            <div className="grid grid-cols-3 gap-1">
                {[0, 1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
                    <div
                        key={i}
                        className={`${size === 'sm' ? 'w-2 h-2' : size === 'lg' ? 'w-4 h-4' : size === 'xl' ? 'w-5 h-5' : 'w-3 h-3'} rounded animate-pulse`}
                        style={{
                            backgroundColor: colorValue,
                            animationDelay: `${i * 0.1}s`,
                            animationDuration: '1.2s'
                        }}
                    />
                ))}
            </div>
        )
    };

    const loaderContent = (
        <div className={`flex flex-col items-center justify-center ${currentSize.gap} ${className}`}>
            {loaders[type] || loaders.spinner}
            {text && (
                <p className={`${currentSize.text} text-gray-600 font-medium`}>
                    {text}
                </p>
            )}
        </div>
    );

    if (fullScreen) {
        return (
            <div className="fixed inset-0 z-50 flex items-center justify-center backdrop-blur-sm bg-black/30">
                <div className="bg-white rounded-lg p-6 shadow-xl">
                    {loaderContent}
                </div>
            </div>
        );
    }

    return loaderContent;
};


export default Loader;  