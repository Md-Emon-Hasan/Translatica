import React from 'react';

interface MainLayoutProps {
    children: React.ReactNode;
}

export const MainLayout: React.FC<MainLayoutProps> = ({ children }) => {
    return (
        <div className="bg-cover bg-no-repeat bg-center bg-fixed min-h-screen relative flex flex-col font-sans text-text overflow-x-hidden"
            style={{ backgroundImage: "url('https://images.unsplash.com/photo-1501004318641-b39e6451bec6?ixlib=rb-4.0.3&auto=format&fit=crop&w=1950&q=80')" }}>

            {/* Overlay */}
            <div className="absolute inset-0 z-0 pointer-events-none bg-gradient-to-b from-[#080810]/65 via-[#0f0f1a]/55 to-[#080810]/70" />

            {/* Content */}
            <div className="relative z-10 flex-1 flex flex-col items-center p-5 md:p-10 max-w-[800px] mx-auto w-full">
                {children}
            </div>

        </div>
    );
};
