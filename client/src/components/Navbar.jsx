import React from 'react';
import { FileText, Github } from 'lucide-react';
import { Link } from 'react-router-dom';

const Navbar = () => {
    return (
        <nav className='sticky top-0 z-50 glass border-b border-slate-200/50'>
            <div className='max-w-7xl mx-auto px-4 sm:px-6 lg:px-8'>
                <div className='flex justify-between items-center h-16'>
                    {/* Logo */}
                    <Link to="/" className='flex items-center gap-2 group'>
                        <div className="bg-[var(--color-primary)] p-2 rounded-lg text-white group-hover:scale-105 transition-transform duration-200">
                            <FileText size={24} />
                        </div>
                        <span className='font-bold text-xl tracking-tight text-[var(--color-text-main)]'>
                            DocClassify
                        </span>
                    </Link>

                    {/* Navigation */}
                    <div className='flex items-center gap-6'>
                        <a
                            href="#"
                            className='text-[var(--color-text-muted)] hover:text-[var(--color-primary)] transition-colors font-medium text-sm'
                        >
                            Documentation
                        </a>
                        <a
                            href="#"
                            className='text-[var(--color-text-muted)] hover:text-[var(--color-primary)] transition-colors font-medium text-sm'
                        >
                            About
                        </a>
                        <a
                            href="https://github.com"
                            target="_blank"
                            rel="noreferrer"
                            className='text-[var(--color-text-muted)] hover:text-[var(--color-text-main)] transition-colors'
                        >
                            <Github size={20} />
                        </a>
                    </div>
                </div>
            </div>
        </nav>
    )
}

export default Navbar;
