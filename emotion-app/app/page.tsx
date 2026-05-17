import Navbar from "@/components/Navbar";
import HeroSection from "@/components/HeroSection";
import StatsSection from "@/components/StatsSection";
import AboutSection from "@/components/AboutSection";
import EventsSection from "@/components/EventsSection";
import ReviewsSection from "@/components/ReviewsSection";
import PricingSection from "@/components/PricingSection";
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
        <ReviewsSection />
        <PricingSection />
        <FAQ />
        <Footer />
      </main>
    </>
  );
}
