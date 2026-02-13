import { useEffect, useRef } from 'react';

export const useParticles = () => {
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const container = containerRef.current;
        if (!container) return;

        const createParticle = () => {
            const particle = document.createElement('div');
            particle.className = 'absolute rounded-full opacity-30 bottom-[-10px] animate-[floatUp_linear_infinite]';

            // Random properties to match original generic CSS
            // The original CSS used :nth-child for colors, but we'll inline styles here or use classes
            // We will rely on random assignment of classes for colors

            const size = Math.random() * 3 + 2; // 2px to 5px
            const colors = ['bg-primary', 'bg-secondary', 'bg-accent'];
            const colorClass = colors[Math.floor(Math.random() * colors.length)];

            particle.classList.add(colorClass);
            particle.style.width = `${size}px`;
            particle.style.height = `${size}px`;
            particle.style.left = `${Math.random() * 100}%`;
            particle.style.animationDelay = `${Math.random() * 20}s`;
            particle.style.animationDuration = `${15 + Math.random() * 20}s`;

            container.appendChild(particle);

            // Cleanup isn't strictly necessary for a fixed set of particles that loop, 
            // but in React we might want to avoid duplicates if effect re-runs.
            // However, the original script just added 50 particles once.
        };

        // Clear existing particles to prevent duplication on re-renders (strict mode)
        container.innerHTML = '';

        for (let i = 0; i < 50; i++) {
            createParticle();
        }

        // Add keyframes style if not exists (Tailwind config should handle this, 
        // but specific keyframes were in original CSS)
        // We added 'floatUp' to tailwind config? No wait, we need to check tailwind config.
        // I added 'fadeInDown' etc, but not 'floatUp'. I should add it to global CSS or Tailwind config.

    }, []);

    return containerRef;
};
