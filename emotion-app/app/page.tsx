import Navbar from "@/components/Navbar";
import HeroSection from "@/components/HeroSection";
import StatsSection from "@/components/StatsSection";
import AboutSection from "@/components/AboutSection";
import EventsSection from "@/components/ArchitectureSection";
import ReviewsSection from "@/components/ResultsSection";
import PricingSection from "@/components/ComparisonSection";
import FAQ from "@/components/FAQ";
import Footer from "@/components/Footer";
import AudioSection from "@/components/AudioSection";
import VideoSection from "@/components/VideoSection";
import CsvSection from "@/components/CsvSection";

export default function Home() {
  return (
    <>
      <Navbar />
      <main>
        <HeroSection />
        <StatsSection />
        <AboutSection />
        <AudioSection />
        <VideoSection />
        <CsvSection />
        <EventsSection />
        <PricingSection />
        <FAQ />
        <Footer />
      </main>
    </>
  );
}
