import React from 'react';

const Footer = () => {
    return (
        <footer className='border-t border-slate-200 bg-white py-8'>
            <div className='max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row justify-between items-center gap-4'>
                <p className='text-sm text-[var(--color-text-muted)]'>
                    &copy; {new Date().getFullYear()} DocClassify. All rights reserved.
                </p>
                <div className='flex gap-6'>
                    <a href="#" className='text-sm text-[var(--color-text-muted)] hover:text-[var(--color-primary)] transition-colors'>
                        Privacy Policy
                    </a>
                    <a href="#" className='text-sm text-[var(--color-text-muted)] hover:text-[var(--color-primary)] transition-colors'>
                        Terms of Service
                    </a>
                </div>
            </div>
        </footer>
    )
}

export default Footer;
